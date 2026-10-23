"""Read-only live suitable selection; later screening databases take precedence."""
import sqlite3
import time
from pathlib import Path


class SuitableSelection:
    def __init__(self,databases,refresh_seconds=10):
        self.databases=[Path(p) for p in databases]
        self.refresh_seconds=refresh_seconds
        self.updated=0.0
        self.rows={}

    def refresh(self,force=False):
        if not force and time.monotonic()-self.updated<self.refresh_seconds:return self.rows
        selected={}
        for path in self.databases:
            if not path.is_file():continue
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=5) as connection:
                columns={row[1] for row in connection.execute('PRAGMA table_info(tasks)')}
                if not {'id','family','package','status','decision'}.issubset(columns):continue
                # A newer completed negative/uncertain decision overrides an older suitable.
                existing=list(selected)
                for start in range(0,len(existing),400):
                    ids=existing[start:start+400]
                    placeholders=','.join('?' for _ in ids)
                    for ident, in connection.execute(f"SELECT id FROM tasks WHERE status='succeeded' AND id IN ({placeholders})",ids):
                        selected.pop(ident,None)
                for ident,family,package in connection.execute("SELECT id,family,package FROM tasks WHERE status='succeeded' AND decision='suitable'"):
                    selected[ident]=dict(id=ident,family=family,package=package)
        self.rows=selected;self.updated=time.monotonic()
        return self.rows
