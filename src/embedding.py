
import torch
from imagebind.models.imagebind_model import ModalityType
import numpy as np
import torch
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from imagebind.data import get_clip_timepoints, SpatialCrop
from typing import List
from collections import defaultdict
from functools import lru_cache
from decord import VideoReader, cpu, bridge
import math
import time

bridge.set_bridge("torch")  # decord returns torch tensors

def checkOverlapCondition(st_time, end_time, ref_times=[0, 1e9]):
    return ref_times[0] <= st_time <= ref_times[1] or st_time<=ref_times[0] <= end_time

def get_tags_interval(all_tags, start_time: int, end_time: int, padding: int, include_tracks: List[str]):
    res = defaultdict(list)
    ref_times = [max(start_time-padding, 0), end_time+padding]
    # print(ref_times)
    for chunk in all_tags:
        if chunk['label'] != 'Shot Tags':
            return
        for tag in chunk['tags']:
            # print(tag["start_time"], tag['end_time'], checkOverlapCondition(tag["start_time"], tag['end_time'], ref_times=ref_times))
            if checkOverlapCondition(tag["start_time"], tag['end_time'], ref_times=ref_times):
                tag_fields = tag["text"]
                for track, track_data in tag_fields.items():
                    if track not in include_tracks:
                        continue
                    for td in track_data:
                        if checkOverlapCondition(td["start_time"], td["end_time"], ref_times=ref_times):
                            res[track].append(td)
    return res

def processandcompressdwnldTagsInterval(tracks = ['Powermove'], start_time=0, end_time=1e6, all_tags = None):
    # print('Processing Tags (No Download)')
    all_tags_ts = get_tags_interval(all_tags, start_time, end_time, 0, tracks)

    all_tag_data = dict()
    for track in tracks:
        if track in all_tags_ts:
            all_tag_data[track] = []
            for td in all_tags_ts[track]:
                td_txt = {'start_time':td['start_time'], 'end_time': td['end_time'], 'text': td['text']} 
                all_tag_data[track].append(td_txt)   
    return  all_tag_data



def getMoves(text):
    entities = text.split(';')
    movedict = {}
    for ent in entities:
        tmp = ent.split(':')
        movedict[tmp[0].strip()] = tmp[1].strip()
    returnlab = ''
    if 'Category' in movedict:returnlab+=movedict['Category']
    if 'Move' in movedict:returnlab+=':'+movedict['Move']
    if 'Variation' in movedict: returnlab+=':'+movedict['Variation']
    return returnlab


def getallMovesfromTags(seg_tags, seltracks = None):
    retlabel = []
    if seltracks == None: seltracks = list(seg_tags.keys())

    for track in seg_tags:
        if track in seltracks:
            for move in seg_tags[track]:
                retlabel.append(getMoves(move['text'][0]))

    return retlabel




def getipseglevel(video_path, all_tags, 
                    start_time=0, 
                    clip_duration = 2, 
                    clips_per_video =20, 
                    stride = 5, device = torch.device('cuda'), 
                    verbose = 0, 
                    allfields = ['Backrocks', 'Flips&Jumps','Footwork', 'Freeze','Godown', 'Godowns','Powermove', 'Powermoves', 'Toprock', 'Windmill Only']): # Assume one vid has one label
    
    '''This function creates input tensors for video segments of duration 5s.'''
    
    labels = []
    video_outputs = []
    labelsdict = []
    # Get the Video
    video = EncodedVideo.from_path(
                video_path,
                decoder="decord",
                decode_audio=False,
                **{"sample_rate": 16000},
            )

    # Tranformation as per imagebind source code!
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    dur_full = video.duration    
    end_time = start_time+dur_full
    dur = int(dur_full)

    if verbose>1: print('The total Clip Duration ={}'.format(dur_full))

    for st in range(0, dur, stride):
        end = min(st+stride,dur)
        if verbose>1: print('Processing chunk [{}, {}] from {}'.format(st, end, dur))

        global_st_ms = (start_time+st)*1000
        global_end_ms = (start_time+end)*1000
        seg_tags = processandcompressdwnldTagsInterval(tracks = allfields, start_time=global_st_ms, end_time=global_end_ms, all_tags = all_tags)
        
        retlabs = getallMovesfromTags(seg_tags, seltracks = allfields)
        if len(retlabs) == 0: retlabs = ['None']
        if verbose>1:print('The Label in segment [{}, {}] = {}'.format(global_st_ms, global_end_ms, retlabs))

        clip = video.get_clip(st, end)
        clip_seg_dur = end-st

        if clip_seg_dur<=clip_duration: clip_duration_var = 0.5*clip_seg_dur
        else: clip_duration_var =clip_duration

        
        if clip is None: raise ValueError("No clip found")
        
        clip_sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration_var, clips_per_video=clips_per_video)
        frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)
        all_clips_timepoints = get_clip_timepoints(clip_sampler, clip_seg_dur)

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            st_pnt = st+clip_timepoints[0]
            ed_pnt = st+clip_timepoints[1]

            if verbose>1: print(st_pnt, ed_pnt)
            clip = video.get_clip(st_pnt,ed_pnt)
            
            if clip is None: raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1
            # print(video_clip.shape)
            all_video.append(video_clip)

        all_video = [video_transform(clip) for clip in all_video]
        all_video = SpatialCrop(224, num_crops=3)(all_video)
        # print('After Video Transform')
        # print('Len = {}, shape = {}'.format(len(all_video), all_video[0].shape))

        all_video = torch.stack(all_video, dim=0).to('cpu')
        video_outputs.append(all_video)
        labels.append(retlabs)  ## Will change for part level
        labelsdict.append({'start_time': global_st_ms,'end_time': global_end_ms,'labels':retlabs})

    if len(video_outputs)>0: result = torch.stack(video_outputs, dim=0).to('cpu')
    else: result, labels = None, None
    return result, labels, labelsdict, end_time




def getembedpartpaths(vidfilepaths, all_tags, model = None, verbose = 0): ## Assume same label for entire video!

    all_embed = []
    all_labels = []
    all_labels_dict = []

    partdict, minp, maxp = mappartspath(vidfilepaths)
    start_time = 0
    for i in range(maxp):
        if i in partdict:
            part_path = partdict[i]
            if verbose>0: print('Processing file {}: Start Time = {}'.format(partdict[i], start_time))
            vid_pix_i, y_i, labelsdict, start_time = getipseglevel(part_path, all_tags, 
                    start_time=start_time, 
                    clip_duration = 2, 
                    clips_per_video =20, 
                    stride = 5, device = torch.device('cuda'), 
                    verbose = verbose, 
                    allfields = ['Backrocks', 'Flips&Jumps','Footwork', 'Freeze','Godown', 'Godowns','Powermove', 'Powermoves', 'Toprock'])
            if vid_pix_i is None: continue
            if verbose>0: print('Following Tags exist = {}'.format(y_i))
            all_labels+=y_i
            all_labels_dict+=labelsdict
            vid_pix_i = vid_pix_i.to('cuda')
            event_inputs = {ModalityType.VISION: vid_pix_i,}
            with torch.no_grad(): event_embeddings = model(event_inputs)
            embed_np = event_embeddings['vision'].cpu().numpy()
            if len(all_embed): all_embed = np.vstack((all_embed, embed_np))
            else: all_embed = embed_np
    return all_embed, all_labels, all_labels_dict



def getMoves_mod(text):
    movedict = {}
    for val in text:
        if 'Category' in val: movedict['Category'] = val.split(':')[1]
        if 'Move' in val: movedict['Move'] = val.split(':')[1]
        if 'Variation' in val: movedict['Variation'] = val.split(':')[1]
        if 'Athlete' in val: movedict['Athlete'] = val.split(':')[1]

    returnlab = ''
    if 'Category' in movedict:returnlab+=movedict['Category'].strip()
    if 'Move' in movedict:returnlab+=':'+movedict['Move'].strip()
    if 'Variation' in movedict: returnlab+=':'+movedict['Variation'].strip()
    return returnlab, movedict


def getallMovesfromTags_mod(seg_tags, seltracks = None):
    retlabel = []
    retlabdict = []
    if seltracks == None: 
        seltracks = list(seg_tags.keys())

    for track in seg_tags:
        if track in seltracks:
            for move in seg_tags[track]:
                # print(move['text'])
                mv, mvdict = getMoves_mod(move['text']) if type(move['text']) ==  list else (move['text'], {})
                retlabel.append(mv)
                retlabdict.append(mvdict)

    return retlabel, retlabdict

def get_tags_interval_mod(track_tags, start_time: int, end_time: int, padding: int):
    ref_times = [max(start_time-padding, 0), end_time+padding]
    return [tag for tag in track_tags if checkOverlapCondition(tag["start_time"], tag['end_time'], ref_times=ref_times)]
    

def processandcompressdwnldTagsInterval_mod(seltracks = None, start_time=0, end_time=1e6, all_tags = None):

    if seltracks == None: seltracks = list(all_tags['metadata_tags'].keys())
    
    all_tag_data = dict()
    for track in all_tags['metadata_tags']:
        if track in seltracks:
            track_tag = all_tags['metadata_tags'][track]
            all_tag_data[track_tag['label']] = get_tags_interval_mod(track_tag['tags'], start_time, end_time, 0)
    return  all_tag_data



def mappartspath_mod(vidfilepaths):
    returndict = {}
    for path in vidfilepaths:
        partno = int((path.split('/')[-1]).split('.')[0])
        returndict[partno] = path
    nparts = returndict.keys()
    minp, maxp = min(nparts), max(nparts)
    return returndict, minp, maxp

def getipseglevel_mod(video_path, all_tags, 
                    start_time=0, 
                    clip_duration = 2, 
                    clips_per_video =20, 
                    stride = 5, device = torch.device('cuda'), 
                    verbose = 0, 
                    allfields = None): # Assume one vid has one label
    
    '''This function creates input tensors for video segments of duration 5s.'''
    
    labels = []
    video_outputs = []
    labelsdict = []
    # Get the Video
    video = EncodedVideo.from_path(
                video_path,
                decoder="decord",
                decode_audio=False,
                **{"sample_rate": 16000},
            )

    # Tranformation as per imagebind source code!
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    dur_full = video.duration    
    end_time = start_time+dur_full
    dur = int(dur_full)

    if verbose>1: print('The total Clip Duration ={}'.format(dur_full))

    for st in range(0, dur, stride):
        end = min(st+stride,dur)
        if verbose>1: print('Processing chunk [{}, {}] from {}'.format(st, end, dur))

        global_st_ms = (start_time+st)*1000
        global_end_ms = (start_time+end)*1000
        seg_tags = processandcompressdwnldTagsInterval_mod(seltracks = allfields, start_time=global_st_ms, end_time=global_end_ms, all_tags = all_tags)
        # processandcompressdwnldTagsInterval(tracks = allfields, start_time=global_st_ms, end_time=global_end_ms, all_tags = all_tags)
        retlabs, _ = getallMovesfromTags_mod(seg_tags, seltracks = None)
        if len(retlabs) == 0: retlabs = ['None']
        if verbose>1:print('The Label in segment [{}, {}] = {}'.format(global_st_ms, global_end_ms, retlabs))

        clip = video.get_clip(st, end)
        clip_seg_dur = end-st

        if clip_seg_dur<=clip_duration: clip_duration_var = 0.5*clip_seg_dur
        else: clip_duration_var =clip_duration

        
        if clip is None: raise ValueError("No clip found")
        
        clip_sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration_var, clips_per_video=clips_per_video)
        frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)
        all_clips_timepoints = get_clip_timepoints(clip_sampler, clip_seg_dur)

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            st_pnt = st+clip_timepoints[0]
            ed_pnt = st+clip_timepoints[1]

            if verbose>1: print(st_pnt, ed_pnt)
            clip = video.get_clip(st_pnt,ed_pnt)
            
            if clip is None: raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1
            # print(video_clip.shape)
            all_video.append(video_clip)

        all_video = [video_transform(clip) for clip in all_video]
        all_video = SpatialCrop(224, num_crops=3)(all_video)
        # print('After Video Transform')
        # print('Len = {}, shape = {}'.format(len(all_video), all_video[0].shape))

        all_video = torch.stack(all_video, dim=0).to('cpu')
        video_outputs.append(all_video)
        labels.append(retlabs)  ## Will change for part level
        labelsdict.append({'start_time': global_st_ms,'end_time': global_end_ms,'labels':retlabs})

    if len(video_outputs)>0: result = torch.stack(video_outputs, dim=0).to('cpu')
    else: result, labels = None, None
    return result, labels, labelsdict, end_time


def getipseglevel_fast(
    video_path, all_tags,
    start_time=0, 
    clip_duration=2,
    clips_per_video=20,
    stride=5,
    device=torch.device("cuda"),  # kept for signature parity; processing stays on CPU for determinism
    verbose=0, allfields = None):



    # ---- Reader + basic video info ----
    vr = VideoReader(video_path, ctx=cpu(0))
    T = len(vr)
    avg_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 0.0
    if T == 0 or avg_fps <= 0:
        return None

    # Matches EncodedVideo.get_clip indexing: start_idx = ceil(fps * t)
    def sec_to_idx(t: float) -> int:
        return min(T, int(math.ceil(avg_fps * t)))

    # ---- Transforms (identical order to original) ----
    video_transform = transforms.Compose([
        pv_transforms.ShortSideScale(224),
        NormalizeVideo(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

    # Helper: get the exact temporal subsample indices this transform will pick for a window of length L
    @lru_cache(maxsize=None)
    def subsample_indices_for_len(L: int) -> torch.Tensor:
        # Create a tiny dummy clip whose "pixel values" equal their time index; run the transform to read back indices.
        dummy = torch.arange(L, dtype=torch.float32).view(1, L, 1, 1)  # (C=1,T=L,H=1,W=1)
        picked = frame_sampler(dummy)[0, :, 0, 0].to(torch.int64)      # shape: (clip_duration,)
        return picked  # CPU tensor of indices (may contain duplicates when L < clip_duration)

    # ---- First pass: figure out ALL absolute frame indices needed across the whole video ----
    dur_full = T / avg_fps
    dur = int(dur_full)

    needed_abs_idxs = set()
    windows = []  # keep metadata to rebuild later: (i0_abs, i1_abs, subsample_rel_indices)
    for st_s in range(0, dur, stride):
        ed_s = min(st_s + stride, dur)
        i0_abs, i1_abs = sec_to_idx(st_s), sec_to_idx(ed_s)
        if i1_abs <= i0_abs:
            continue

        seg_dur = ed_s - st_s
        clip_duration_var = 0.5 * seg_dur if seg_dur <= clip_duration else clip_duration
        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration_var, clips_per_video=clips_per_video
        )
        all_clips_timepoints = get_clip_timepoints(clip_sampler, seg_dur)

        for rel_start_s, rel_end_s in all_clips_timepoints:
            abs_rs = sec_to_idx(st_s + rel_start_s)
            abs_re = sec_to_idx(st_s + rel_end_s)
            if abs_re <= abs_rs:
                continue

            L = abs_re - abs_rs
            rel_pick = subsample_indices_for_len(L)   # (clip_duration,)
            abs_pick = (abs_rs + rel_pick).tolist()
            needed_abs_idxs.update(abs_pick)

            # stash for reconstruction
            windows.append((abs_rs, abs_re, rel_pick))

    if not needed_abs_idxs:
        return None

    # ---- Decode exactly those frames once ----
    abs_idx_sorted = sorted(needed_abs_idxs)
    frames_sel = vr.get_batch(abs_idx_sorted)        # (Nsel,H,W,C) uint8 torch
    # Reorder to (C,Nsel,H,W) for fast gather by time
    frames_sel = frames_sel.permute(3, 0, 1, 2).contiguous()  # uint8
    idx_to_pos = {idx: pos for pos, idx in enumerate(abs_idx_sorted)}

    # ---- Second pass: rebuild each window's subsampled clip from cached frames, then transform ----
    video_outputs = []
    labelsdict = []
    labels = []   
    end_time = start_time+dur_full

    for st_s in range(0, dur, stride):
        ed_s = min(st_s + stride, dur)

        if verbose>1: print('Processing chunk [{}, {}] from {}'.format(st_s, ed_s, dur))

        global_st_ms = (start_time+st_s)*1000
        global_end_ms = (start_time+ed_s)*1000
        seg_tags = processandcompressdwnldTagsInterval_mod(seltracks = allfields, start_time=global_st_ms, end_time=global_end_ms, all_tags = all_tags)
        retlabs, _ = getallMovesfromTags_mod(seg_tags, seltracks = None)
        if len(retlabs) == 0: retlabs = ['None']
        if verbose>1:print('The Label in segment [{}, {}] = {}'.format(global_st_ms, global_end_ms, retlabs))


        i0_abs, i1_abs = sec_to_idx(st_s), sec_to_idx(ed_s)
        if i1_abs <= i0_abs:
            continue

        seg_dur = ed_s - st_s
        clip_duration_var = 0.5 * seg_dur if seg_dur <= clip_duration else clip_duration
        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration_var, clips_per_video=clips_per_video
        )
        all_clips_timepoints = get_clip_timepoints(clip_sampler, seg_dur)

        all_video = []
        if verbose:
            print(f"Number of clips to be sampled = {len(all_clips_timepoints)}")

        for rel_start_s, rel_end_s in all_clips_timepoints:
            abs_rs = sec_to_idx(st_s + rel_start_s)
            abs_re = sec_to_idx(st_s + rel_end_s)
            if abs_re <= abs_rs:
                continue

            L = abs_re - abs_rs
            rel_pick = subsample_indices_for_len(L)  # (clip_duration,)
            # Map absolute frame indices to cached positions
            pos_idx = [idx_to_pos[abs_rs + int(i)] for i in rel_pick]
            window = frames_sel[:, pos_idx, ...]      # (C, clip_duration, H, W) uint8

            # Apply the SAME order as original: subsample (we already did), then scale, then spatial transforms.
            clip_t = window.to(torch.float32) / 255.0
            clip_t = video_transform(clip_t)          # ShortSideScale -> Normalize
            # 3-crop expansion
            crops = SpatialCrop(224, num_crops=3)([clip_t])  # list[Tensor]
            all_video.extend(crops)

        if not all_video:
            continue

        batch = torch.stack(all_video, dim=0).to("cpu")
        video_outputs.append(batch)
        labels.append(retlabs)  ## Will change for part level
        labelsdict.append({'start_time': global_st_ms,'end_time': global_end_ms,'labels':retlabs})

    if len(video_outputs)>0: result = torch.stack(video_outputs, dim=0).to('cpu')
    else: result, labels = None, None
    return result, labels, labelsdict, end_time

def getembedpartpaths_mod(vidfilepaths, all_tags, model = None, verbose = 0, fast = False): ## Assume same label for entire video!

    all_embed = []
    all_labels = []
    all_labels_dict = []

    partdict, minp, maxp = mappartspath_mod(vidfilepaths)
    if verbose: print('Parts: Min = {}, Max = {}, partdict = {}'.format(minp,maxp,partdict))
    start_time = 0
    for i in range(maxp):
        if i in partdict:
            part_path = partdict[i]
            if verbose>0: print('Processing file {}: Start Time = {}'.format(partdict[i], start_time))
            if fast:
                
                st_time = time.time()

                vid_pix_i, y_i, labelsdict, start_time = getipseglevel_fast(part_path, all_tags, 
                        start_time=start_time, 
                        clip_duration = 2, 
                        clips_per_video =20, 
                        stride = 5, device = torch.device('cuda'), 
                        verbose = verbose, 
                        allfields = None)

                e_time = time.time()
                elapsed_time = e_time - st_time
                print(f"Time FAST part {i}: {elapsed_time:.4f} seconds")

            else:
                st_time = time.time()

                vid_pix_i, y_i, labelsdict, start_time = getipseglevel_mod(part_path, all_tags, 
                        start_time=start_time, 
                        clip_duration = 2, 
                        clips_per_video =20, 
                        stride = 5, device = torch.device('cuda'), 
                        verbose = verbose, 
                        allfields = None)
                
                e_time = time.time()
                elapsed_time = e_time - st_time
                print(f"Time ORG part {i}: {elapsed_time:.4f} seconds")

            if vid_pix_i is None: continue
            if verbose>0: print('Following Tags exist = {}'.format(y_i))
            all_labels+=y_i
            all_labels_dict+=labelsdict
            vid_pix_i = vid_pix_i.to('cuda')
            event_inputs = {ModalityType.VISION: vid_pix_i,}
            with torch.no_grad(): event_embeddings = model(event_inputs)
            embed_np = event_embeddings['vision'].cpu().numpy()
            if len(all_embed): all_embed = np.vstack((all_embed, embed_np))
            else: all_embed = embed_np
    return all_embed, all_labels, all_labels_dict