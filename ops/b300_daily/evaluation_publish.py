"""Bounded publication of closed scoring artifacts; never overwrite a result."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import time


def publish_closed_tree(source, destination, progress, *, workers=8):
    source,destination=Path(source),Path(destination)
    if workers != 8: raise ValueError('Unreviewed publication concurrency')
    files=sorted(path for path in source.rglob('*') if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise ValueError('Expected a nonempty, regular closed artifact tree')
    # One metadata inventory, no repeated parent mkdir/stat for every tiny file.
    jobs=[(path,path.relative_to(source),path.stat().st_size) for path in files]
    destination.mkdir(parents=True,exist_ok=False)
    parents={parent for _,relative,_ in jobs for parent in relative.parents if parent!=Path('.')}
    for parent in sorted(parents,key=lambda item:(len(item.parts),str(item))):
        (destination/parent).mkdir()
    def copy(job):
        path,relative,size=job; final=destination/relative
        with path.open('rb') as inp,final.open('xb') as out:
            shutil.copyfileobj(inp,out,8*1024**2)
            out.flush();os.fsync(out.fileno())
        if final.stat().st_size!=size:
            raise ValueError('Result publication size differs: '+str(final))
    start=time.monotonic();last_update=start;done=0
    progress(published_files_so_far=0,publication_total_files=len(jobs),publication_workers=workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(copy,job) for job in jobs]):
            future.result();done+=1
            if time.monotonic()-last_update>=30 or done==len(jobs):
                progress(published_files_so_far=done,publication_total_files=len(jobs),
                    publication_elapsed_seconds=round(time.monotonic()-start,1))
                last_update=time.monotonic()
    return done
