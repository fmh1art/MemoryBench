"""24 PTS-based samples in a 4-row by 6-column contact sheet, using CPU only."""
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

VERSION='uniform-24-4rows-6cols-v1'


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def make_sheet(video,directory):
    video=Path(video);directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    image_path=directory/'sheet.jpg';meta_path=directory/'sheet.json'
    st=video.stat()
    identity=dict(version=VERSION,video=str(video.resolve()),source_bytes=st.st_size,source_mtime_ns=st.st_mtime_ns)
    if image_path.is_file() and meta_path.is_file():
        meta=json.loads(meta_path.read_text())
        if all(meta.get(k)==v for k,v in identity.items()) and hashlib.sha256(image_path.read_bytes()).hexdigest()==meta['image_sha256']:
            return meta
    width,height,label,header=384,292,28,44
    canvas=Image.new('RGB',(6*width,4*(height+label)+header),'#151a23')
    draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=18)
    actual=[];pts=[]
    with av.open(str(video)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        if stream.duration is None:raise ValueError('No video duration in MP4 header')
        duration=float(stream.duration*stream.time_base)
        if duration<=0:raise ValueError('Nonpositive video duration')
        origin=float((stream.start_time or 0)*stream.time_base)
        fps=float(stream.average_rate or stream.guessed_rate or 30)
        targets=np.linspace(0,max(0,duration-1/fps),24).tolist()
        draw.text((12,10),f'24 uniform samples | 4 rows x 6 columns | video {duration:.3f}s | chronological left-to-right',fill='white',font=font)
        for index,target in enumerate(targets):
            container.seek(int((origin+target)/stream.time_base),stream=stream,backward=True)
            previous=None;chosen=None
            for frame in container.decode(stream):
                if frame.pts is None:raise ValueError('Decoded frame has no PTS')
                stamp=float(frame.pts*stream.time_base)-origin
                if stamp>=target:
                    chosen=(stamp,frame)
                    if previous and abs(previous[0]-target)<=abs(stamp-target):chosen=previous
                    break
                previous=(stamp,frame)
            if chosen is None:chosen=previous
            if chosen is None:raise ValueError(f'No frame decoded for t={target}')
            stamp,frame=chosen
            thumb=frame.reformat(width=width,height=max(1,round(frame.height*width/frame.width))).to_image()
            thumb=ImageOps.contain(thumb,(width-4,height-4),Image.Resampling.LANCZOS)
            x,y=(index%6)*width,header+(index//6)*(height+label)
            canvas.paste(thumb,(x+(width-thumb.width)//2,y+(height-thumb.height)//2))
            draw.text((x+8,y+height+3),f'#{index+1:02d}  t={stamp:.3f}s',fill='white',font=font)
            actual.append(stamp);pts.append(frame.pts)
        time_base=[stream.time_base.numerator,stream.time_base.denominator]
        frame_count=stream.frames
    tmp=image_path.with_suffix('.tmp.jpg');canvas.save(tmp,quality=90,subsampling=0);tmp.replace(image_path)
    meta=dict(identity,image_path=str(image_path.resolve()),image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
              duration_seconds=duration,source_frame_count=frame_count,time_base=time_base,
              target_seconds=targets,actual_seconds=actual,actual_pts=pts,unique_frames=len(set(pts)),
              layout_rows=4,layout_columns=6,image_size=list(canvas.size))
    atomic_json(meta_path,meta)
    return meta
