"""Fork an existing run for stricter occlusion review without regenerating descriptions."""
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import zlib

from .run import MODEL,POLICY_VERSION,build_config,digest,now
from .sheets import atomic_json


def fork_for_occlusion(previous_database,previous_output,database,output):
    previous_database=Path(previous_database);database=Path(database);output=Path(output)
    if database.exists():raise FileExistsError(f'Existing queue must be resumed, not overwritten: {database}')
    database.parent.mkdir(parents=True,exist_ok=True)
    temporary=database.with_suffix('.migrating.sqlite3')
    if temporary.exists():raise FileExistsError(f'Incomplete earlier migration: {temporary}')
    with previous_database.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with sqlite3.connect(previous_database.resolve().as_uri()+'?mode=ro',uri=True) as source:
            db=sqlite3.connect(temporary);source.backup(db)
        columns={r[1] for r in db.execute('PRAGMA table_info(tasks)')}
        if 'prior_result_path' not in columns:db.execute('ALTER TABLE tasks ADD COLUMN prior_result_path TEXT')
        previous_config=json.loads(db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()[0])
        recovered=0
        for ident,family,package in db.execute("SELECT id,family,package FROM tasks WHERE status='running'").fetchall():
            path=Path(previous_output)/family/package[:2]/package/'result.json'
            if not path.is_file():continue
            result=json.loads(path.read_text())
            if result.get('config_sha256')!=digest(previous_config):continue
            usage={k:sum((t.get('usage') or {}).get(k,0) or 0 for t in result['request_traces']) for k in ('prompt_tokens','completion_tokens','total_tokens')}
            db.execute("UPDATE tasks SET status='succeeded',decision=?,result_path=?,result_zlib=?,prompt_tokens=?,completion_tokens=?,total_tokens=? WHERE id=?",
                (result['annotation']['memorybench']['decision'],str(path),zlib.compress(json.dumps(result,ensure_ascii=False).encode()),usage['prompt_tokens'],usage['completion_tokens'],usage['total_tokens'],ident))
            recovered+=1
        retained=0
        for ident,raw in db.execute("SELECT id,result_zlib FROM tasks WHERE status='succeeded' AND decision!='suitable'").fetchall():
            result=json.loads(zlib.decompress(raw))
            result['gate_reuse']=dict(policy_version=POLICY_VERSION,reason='Prior non-suitable decision retained under stricter admission rule; no new model call',source_database=str(previous_database))
            db.execute('UPDATE tasks SET result_zlib=? WHERE id=?',(zlib.compress(json.dumps(result,ensure_ascii=False).encode()),ident));retained+=1
        review=db.execute("SELECT count(*) FROM tasks WHERE status='succeeded' AND decision='suitable'").fetchone()[0]
        db.execute("UPDATE tasks SET status='pending',prior_result_path=result_path,result_path=NULL,result_zlib=NULL,decision=NULL,prompt_tokens=0,completion_tokens=0,total_tokens=0,error=NULL,attempts=0 WHERE status='succeeded' AND decision='suitable'")
        db.execute("UPDATE tasks SET status='pending',error=NULL WHERE status IN ('running','failed')")
        config=build_config(previous_config.get('model',MODEL))
        db.execute("UPDATE metadata SET value=? WHERE key='config'",(json.dumps(config,sort_keys=True),))
        summary=dict(created_at=now(),source_database=str(previous_database),policy_version=POLICY_VERSION,
            cached_results_recovered=recovered,descriptions_retained_without_call=retained,suitable_to_recheck=review,
            counts=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status')))
        db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',('occlusion_migration',json.dumps(summary)))
        db.commit();db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
        temporary.replace(database)
    atomic_json(output/'migration.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-database',type=Path,default=Path('/var/lib/fmh/runtime/lightwheel-source-vlm/tasks-v2.sqlite3'))
    p.add_argument('--previous-output',type=Path,default=Path('/mnt/bos/tm9b1x/fmh/data/MemoryBenchAnnotationsV2/lightwheel/source_vlm/ark-20261023-v2'))
    p.add_argument('--database',type=Path,default=Path('/var/lib/fmh/runtime/lightwheel-source-vlm/tasks-v3.sqlite3'))
    p.add_argument('--output-root',type=Path,default=Path('/mnt/bos/tm9b1x/fmh/data/MemoryBenchAnnotationsV2/lightwheel/source_vlm/ark-20261023-v3'))
    a=p.parse_args();print(json.dumps(fork_for_occlusion(a.previous_database,a.previous_output,a.database,a.output_root),ensure_ascii=False))


if __name__=='__main__':main()
