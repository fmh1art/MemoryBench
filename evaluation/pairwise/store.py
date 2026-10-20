"""Transactional assignments, stable blinding, private sessions and saved drafts."""

import contextlib
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from pathlib import Path

from . import ANNOTATION_PROTOCOL, CHOICES
from .manifest import split

LEASE_SECONDS = 1800


class Conflict(ValueError):
    pass


def normalize_answers(value, submit=False):
    if set(value) - {"choice", "elapsed_seconds"}:
        raise ValueError("每题只接受一个 A / B / Tie 选择；请刷新页面")
    choice = value.get("choice")
    if (submit or choice is not None) and choice not in CHOICES:
        raise ValueError("请选择 A、B 或 Tie")
    elapsed = value.get("elapsed_seconds", 0)
    if not isinstance(elapsed, (int, float)) or not 0 <= elapsed <= 1e8:
        raise ValueError("Invalid elapsed time")
    return {"choice": choice, "elapsed_seconds": elapsed}


def normalize_vote(value):
    return normalize_answers(value, submit=True)


class Store:
    def __init__(self, path, study, signature, tasks):
        self.path, self.study, self.signature, self.tasks = Path(path), study, signature, tasks
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS raters(
                    id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, secret_hash TEXT NOT NULL UNIQUE,
                    created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS assignments(
                    id TEXT PRIMARY KEY, rater TEXT NOT NULL, task TEXT NOT NULL,
                    swapped INTEGER NOT NULL, state TEXT NOT NULL DEFAULT 'assigned',
                    lease REAL NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                    draft TEXT NOT NULL DEFAULT '{}', revision INTEGER NOT NULL DEFAULT 0,
                    vote TEXT, issue TEXT, UNIQUE(rater, task),
                    FOREIGN KEY(rater) REFERENCES raters(id));
                CREATE INDEX IF NOT EXISTS task_assignments ON assignments(task, state, lease);
                CREATE INDEX IF NOT EXISTS rater_assignments ON assignments(rater, state);
            """)
        with self.transaction() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
            if row and row[0] != signature:
                raise ValueError("Study or input file fingerprints changed; use a new database")
            if not row:
                db.executemany("INSERT INTO metadata VALUES (?,?)", [
                    ("signature", signature), ("secret", secrets.token_hex(32)),
                    ("study", study.model_dump_json()),
                ])
            protocol = db.execute("SELECT value FROM metadata WHERE key='annotation_protocol'").fetchone()
            if protocol and protocol[0] != ANNOTATION_PROTOCOL:
                raise ValueError("Different annotation protocol; use a new database")
            if not protocol:
                # Gate-derived legacy votes are not equivalent to a direct human choice.
                if db.execute("SELECT COUNT(*) FROM assignments WHERE state='voted'").fetchone()[0]:
                    raise ValueError("Legacy Gate/dimension votes cannot be mixed with A/B/Tie; preserve this database and create a new one")
                db.execute("INSERT INTO metadata VALUES ('annotation_protocol', ?)", (ANNOTATION_PROTOCOL,))
                db.execute("UPDATE assignments SET draft='{}',revision=revision+1 WHERE state='assigned'")
            self.secret = db.execute("SELECT value FROM metadata WHERE key='secret'").fetchone()[0]
        self.path.chmod(0o600)

    @contextlib.contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

    @contextlib.contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def session(self, token):
        if not token:
            return None
        with self.connection() as db:
            row = db.execute("SELECT id,name FROM raters WHERE secret_hash=?",
                             (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
            return dict(row) if row else None

    def register(self, name, resume_code=None):
        if resume_code:
            user = self.session(resume_code)
            if user is None:
                raise ValueError("续评码无效")
            return user, resume_code
        name = name.strip()
        if not 1 <= len(name) <= 60:
            raise ValueError("请填写 1–60 字的评估者代号")
        token = secrets.token_urlsafe(32)
        user = {"id": secrets.token_hex(16), "name": name}
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM raters WHERE name=?", (name,)).fetchone():
                raise Conflict("代号已存在，请使用续评码或换一个代号")
            db.execute("INSERT INTO raters VALUES (?,?,?,?)", (
                user["id"], name, hashlib.sha256(token.encode()).hexdigest(), time.time()))
        return user, token

    def random_key(self, *values):
        return hmac.new(self.secret.encode(), json.dumps(values).encode(), hashlib.sha256).hexdigest()

    def assigned(self, db, rater, assignment):
        row = db.execute("SELECT * FROM assignments WHERE id=? AND rater=?", (assignment, rater)).fetchone()
        if row is None:
            raise KeyError("Task not found")
        return row

    def ensure_capacity(self, db, row):
        if row["state"] != "assigned":
            raise Conflict("本题已提交或已报告播放问题")
        active = db.execute("SELECT COUNT(*) FROM assignments WHERE task=? AND id<>? AND (state='voted' OR (state='assigned' AND lease>?))",
                            (row["task"], row["id"], time.time())).fetchone()[0]
        if active >= self.study.raters_per_pair:
            raise Conflict("本题已由其他评估者完成，请进入下一题；草稿仍保留")

    def next(self, rater):
        now = time.time()
        with self.transaction() as db:
            rows = db.execute("SELECT * FROM assignments WHERE rater=?", (rater,)).fetchall()
            occupied = {r["task"] for r in rows if r["state"] != "assigned"}
            counts = {r["task"]: dict(r) for r in db.execute("""
                SELECT task, SUM(state='voted') AS votes,
                SUM(state='voted' OR (state='assigned' AND lease>?)) AS active
                FROM assignments GROUP BY task
            """, (now,))}
            # A still-live assignment is always resumed, including its A/B mapping.
            for row in rows:
                if row["state"] == "assigned" and row["lease"] > now:
                    db.execute("UPDATE assignments SET lease=? WHERE id=?", (now + LEASE_SECONDS, row["id"]))
                    return self.public_task(row), self.progress(db, rater)
            available = [task for key, task in self.tasks.items() if key not in occupied and
                         counts.get(key, {}).get("active", 0) < self.study.raters_per_pair]
            if not available:
                return None, self.progress(db, rater)
            recent = [self.tasks[r["task"]]["case"].case_id for r in sorted(rows, key=lambda r: r["updated"], reverse=True)[:3]]
            previous = {r["task"]: r for r in rows}
            selected = min(available, key=lambda t: (
                t["id"] not in previous,
                t["case"].case_id in recent,
                counts.get(t["id"], {}).get("votes", 0),
                self.random_key(self.study.seed, rater, t["id"]),
            ))
            row = previous.get(selected["id"])
            if row:
                db.execute("UPDATE assignments SET lease=? WHERE id=?", (now + LEASE_SECONDS, row["id"]))
            else:
                aid = secrets.token_hex(16)
                swapped = int(self.random_key(rater, selected["id"], "side"), 16) % 2
                db.execute("INSERT INTO assignments(id,rater,task,swapped,lease,created,updated) VALUES(?,?,?,?,?,?,?)",
                           (aid, rater, selected["id"], swapped, now + LEASE_SECONDS, now, now))
                row = self.assigned(db, rater, aid)
            return self.public_task(row), self.progress(db, rater)

    def progress(self, db, rater):
        completed = db.execute("SELECT COUNT(*) FROM assignments WHERE rater=? AND state='voted'", (rater,)).fetchone()[0]
        issues = db.execute("SELECT COUNT(*) FROM assignments WHERE rater=? AND state='issue'", (rater,)).fetchone()[0]
        total = db.execute("SELECT COUNT(*) FROM assignments WHERE state='voted'").fetchone()[0]
        return {"completed": completed, "issues": issues, "total_tasks": len(self.tasks),
                "study_complete": total >= len(self.tasks) * self.study.raters_per_pair}

    def public_task(self, row):
        task = self.tasks[row["task"]]
        case = task["case"]
        a, b = (task["right"], task["left"]) if row["swapped"] else (task["left"], task["right"])
        clips = {"history": case.history, "gt": case.gt, "A": case.predictions[a], "B": case.predictions[b]}
        return {
            "assignment_id": row["id"], "revision": row["revision"],
            "action_description": case.action_description, "memory_target": case.memory_target,
            "success_criterion": case.success_criterion,
            "read_start_seconds": case.read_start_seconds, "read_end_seconds": case.read_end_seconds,
            "timeline_seconds": case.gt.duration_seconds - case.gt.start_seconds,
            "clips": {slot: {"url": f'/api/media/{row["id"]}/{slot}',
                              "start_seconds": clip.start_seconds,
                              "duration_seconds": clip.duration_seconds} for slot, clip in clips.items()},
            "prompts": {slot: {"recorded": case.predictions[model].prompt_recorded,
                               "text": case.predictions[model].prompt} for slot, model in (("A", a), ("B", b))},
            "draft": json.loads(row["draft"]),
        }

    def heartbeat(self, rater, aid):
        with self.transaction() as db:
            row = self.assigned(db, rater, aid)
            self.ensure_capacity(db, row)
            db.execute("UPDATE assignments SET lease=? WHERE id=?", (time.time() + LEASE_SECONDS, aid))

    def save(self, rater, aid, revision, value, submit=False):
        value = normalize_answers(value, submit)
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        with self.transaction() as db:
            row = self.assigned(db, rater, aid)
            if submit and row["state"] == "voted":
                existing = json.loads(row["vote"])
                if {k: v for k, v in existing.items() if k != "elapsed_seconds"} == {k: v for k, v in value.items() if k != "elapsed_seconds"}:
                    return row["revision"]  # Network retry is idempotent.
                raise Conflict("本题已提交，不能覆盖已有投票")
            self.ensure_capacity(db, row)
            if row["revision"] != revision:
                raise Conflict("另一标签页已更新此题，请刷新后继续")
            now = time.time()
            db.execute("UPDATE assignments SET draft=?,vote=?,state=?,revision=revision+1,updated=?,lease=? WHERE id=?",
                       (encoded, encoded if submit else None, "voted" if submit else "assigned", now, now + LEASE_SECONDS, aid))
            return revision + 1

    def issue(self, rater, aid, description):
        with self.transaction() as db:
            row = self.assigned(db, rater, aid)
            if row["state"] == "issue":
                return
            self.ensure_capacity(db, row)
            db.execute("UPDATE assignments SET state='issue',issue=?,updated=? WHERE id=?", (description, time.time(), aid))

    def media(self, rater, aid, slot):
        with self.connection() as db:
            row = self.assigned(db, rater, aid)
        task = self.tasks[row["task"]]
        case = task["case"]
        if slot == "history":
            return case.history.path
        if slot == "gt":
            return case.gt.path
        if slot not in ("A", "B"):
            raise KeyError("Unknown video")
        a, b = (task["right"], task["left"]) if row["swapped"] else (task["left"], task["right"])
        return case.predictions[a if slot == "A" else b].path

    def records(self):
        with self.connection() as db:
            rows = db.execute("SELECT a.*,r.name AS rater_name FROM assignments a JOIN raters r ON a.rater=r.id WHERE state IN ('voted','issue') ORDER BY a.updated").fetchall()
        records = []
        for row in rows:
            task = self.tasks[row["task"]]
            a, b = (task["right"], task["left"]) if row["swapped"] else (task["left"], task["right"])
            record = {k: row[k] for k in ("id", "rater", "rater_name", "task", "state", "created", "updated", "issue")}
            record.update(case_id=task["case"].case_id, split=split(task["case"].case_id), model_a=a, model_b=b,
                          vote=json.loads(row["vote"]) if row["vote"] else None)
            records.append(record)
        return records
