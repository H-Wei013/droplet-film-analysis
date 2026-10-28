"""Run isolated V5 analyses. All source images and V4 references remain read-only."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import csv,hashlib,json,os,time
import cv2
import numpy as np
import yaml
from droplet_analysis.core import Settings,read_gray,save_image,load_sequence,discover_sequences
from droplet_analysis.v5 import Analyzer,detect_impact
from droplet_analysis.temporal import select_anchors,repair_highlights
from droplet_analysis.render import overlay,sheet
from watercar_tool.review import write_review,write_combined_review
from watercar_tool.io import pixel_csv,series_plots
from watercar_tool.provenance import digest
from droplet_analysis.impact import attach_impact_point
from droplet_analysis.optical import recover_thin_tail,remove_wall_shadow,limit_valid_view,reject_transient_tail

VERSION='5.3.1-confirmed-contact-continuation-and-observed-front'
ROOT=Path(__file__).resolve().parent


def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def analyze_sequence(directory,output,settings_dict,alignment_cache=None,sequence_name=None,cancel_path=None):
    cv2.setNumThreads(1);directory=Path(directory);output=Path(output)
    if output.exists():raise ValueError(f'Output exists; preserve it and choose another run: {output}')
    output.mkdir(parents=True);settings=Settings.from_dict(settings_dict)
    files,numbers,metadata=load_sequence(directory);name=sequence_name or directory.parent.name
    analyzer=Analyzer([read_gray(p) for p in files[:settings.reference_frames]],settings)
    cache_frames=None;cache_provenance=None
    if alignment_cache is not None:
        cache_folder=Path(alignment_cache)/name
        cache_manifest_path=cache_folder/'measurements.json'
        cached=json.loads(cache_manifest_path.read_text(encoding='utf-8'))
        if Path(cached['source_directory']).resolve()!=directory.resolve() or [r['frame'] for r in cached['frames']]!=numbers:raise ValueError('Alignment cache source mismatch')
        for key in ['reference_frames','edge_margin_px','register','roi','wall_points']:
            if cached['settings'][key]!=asdict(settings)[key]:raise ValueError('Alignment cache settings mismatch: '+key)
        if not np.allclose(analyzer.wall.points,np.asarray(cached['wall']['points']),atol=1e-5):raise ValueError('Alignment cache wall mismatch')
        cache_frames={r['frame']:r for r in cached['frames']}
        cache_sources={r['frame']:r['sha256'] for r in json.loads((cache_folder/'source_frames.json').read_text(encoding='utf-8'))}
        cache_provenance=dict(directory=str(cache_folder),manifest_sha256=hashlib.sha256(cache_manifest_path.read_bytes()).hexdigest(),
            original_algorithm_sha256=cached['algorithm_sha256'],raw_hashes_reverified=True,
            alignment_validation_frames=[numbers[j] for j in sorted({0,len(numbers)//2,len(numbers)-1})],
            note='Only lossless aligned images reused. Segmentation, main-film selection, temporal reconstruction, measurement and plots recomputed. Fresh alignment checks must match pixel-for-pixel.')
    source_digest,_=digest();signature=hashlib.sha256((source_digest+str(directory)+json.dumps(asdict(settings),sort_keys=True)).encode()).hexdigest()[:20]
    for sub in ['frames','initial_masks','masks','overlays']:(output/sub).mkdir()
    save_image(output/'background.png',analyzer.background.astype(np.uint8))
    save_image(output/'wall_reference.png',overlay(analyzer.background.astype(np.uint8),analyzer.wall,dict(contours=[],contact_segments=[],maximum_visible_thickness=None)))
    anchors=[];extras=[];sources=[]
    print(name,': segmenting',len(files),'source frames',flush=True)
    for i,(path,number) in enumerate(zip(files,numbers)):
        if cancel_path and Path(cancel_path).exists():raise InterruptedError('Task stopped')
        raw=read_gray(path);raw_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        if cache_frames is not None:
            cached_frame=cache_frames[number]
            if raw_hash!=cache_sources[number]:raise ValueError('Raw source changed since alignment cache')
            aligned=read_gray(cache_folder/cached_frame['image']);translation=cached_frame['registration_translation_px']
            flags=[f for f in cached_frame['flags'] if f.startswith('registration_')]
            if i in {0,len(numbers)//2,len(numbers)-1}:
                fresh,t,alignment_flags=analyzer.align(raw)
                if not np.array_equal(fresh.astype(np.uint8),aligned) or not np.allclose(t,translation,atol=1e-5):raise ValueError('Fresh alignment differs from cache')
        else:aligned,translation,flags=analyzer.align(raw)
        # A shared lossless uint8 alignment stage makes fresh and cached runs
        # feed identical image samples into segmentation.
        aligned=aligned.astype(np.uint8)
        liquid,filled,repaired,delta=analyzer.segment(aligned)
        main,satellites,tracking=analyzer.main_component(liquid,number,aligned)
        anchors.append(dict(frame=number,area_px2=int(main.sum()),repaired_boundary_area_px2=int((repaired&main).sum())))
        extras.append(dict(registration_translation_px=translation,registration_flags=flags,satellite_contours=satellites,
                           main_film_tracking=tracking,associated_fragment_count=analyzer.last_fragment_count,
                           spatial_repair_area_px2=int((repaired&main).sum()),
                           filled_interior_area_px2=int((filled&main).sum()),
                           wall_shadow_removed_area_px2=analyzer.last_shadow_area,front_specular_repair_area_px2=analyzer.last_front_repair_area,
                           spatial_front_extension_removed_area_px2=analyzer.last_front_trim_area))
        sources.append(dict(frame=number,source_file=str(path),sha256=raw_hash))
        save_image(output/'frames'/f'{number:06d}.png',aligned.astype(np.uint8))
        save_image(output/'initial_masks'/f'{number:06d}.png',main.astype(np.uint8)*255)
        if (i+1)%50==0:print(name,'spatial',i+1,'/',len(files),flush=True)
    @lru_cache(maxsize=28)
    def base(i):
        n=numbers[i];return read_gray(output/'frames'/f'{n:06d}.png'),read_gray(output/'initial_masks'/f'{n:06d}.png')>0
    results=[];profiles=[];observed=[];possible=[]
    analyzer.previous_main=None
    print(name,': main-film temporal reconstruction and measurement',flush=True)
    for i,number in enumerate(numbers):
        if cancel_path and Path(cancel_path).exists():raise InterruptedError('Task stopped')
        aligned,initial=base(i);pair=select_anchors(anchors,i,settings.temporal_anchor_radius)
        if pair:
            left,right=pair;fraction=(number-numbers[left])/(numbers[right]-numbers[left]);weights=(1-fraction,fraction)
            main,repair=repair_highlights(analyzer,aligned,initial,[base(left),base(right)],weights)
        else:main,repair=initial.copy(),np.zeros_like(initial);weights=()
        if i>0 and i+1<len(numbers) and numbers[i-1]==number-1 and numbers[i+1]==number+1:
            main,tail_repair=recover_thin_tail(analyzer,aligned,main,[base(i-1)[1],base(i+1)[1]])
            main,tail_removed=reject_transient_tail(analyzer,aligned,main,[base(i-1)[1],base(i+1)[1]])
        else:tail_repair=np.zeros_like(main);tail_removed=np.zeros_like(main)
        main,_,tracking=analyzer.main_component(main,number,aligned)
        main,shadow=remove_wall_shadow(analyzer,aligned,main)
        result,profile,confirmed,potential=analyzer.measure_main(main,aligned,number,repair)
        extra=extras[i];result.update({k:v for k,v in extra.items() if k!='registration_flags'})
        result['flags']+=extra['registration_flags']
        if extra['spatial_repair_area_px2']>20:result['flags'].append('highlight_boundary_reconstructed')
        if extra['associated_fragment_count']>1:result['flags'].append('saturated_main_film_fragments_associated')
        if tracking=='reacquired':result['flags'].append('main_film_reacquired')
        if int((tail_repair&main).sum())>3:result['flags'].append('temporal_thin_tail_recovered')
        if extra['wall_shadow_removed_area_px2']>3:result['flags'].append('wall_shadow_fringe_removed')
        if extra['front_specular_repair_area_px2']>12:result['flags'].append('front_specular_completion')
        if extra['spatial_front_extension_removed_area_px2']+getattr(analyzer,'last_front_trim_area',0)>3:result['flags'].append('unsupported_front_extension_removed')
        if tail_removed.sum()>3:result['flags'].append('transient_weak_tail_removed')
        if not main.any():result['measurement_status']='missing_main_film';result['flags'].append('main_film_missing')
        result.update(source_file=files[i].name,image=f'frames/{number:06d}.png',mask=f'masks/{number:06d}.png',
            temporal_anchor_frames=[numbers[j] for j in pair] if pair else [],temporal_anchor_weights=list(weights),
            temporal_repair_area_px2=int((repair&main).sum()),temporal_tail_repair_area_px2=int((tail_repair&main).sum()),
            transient_tail_removed_area_px2=int(tail_removed.sum()),front_extension_removed_area_px2=extra['spatial_front_extension_removed_area_px2']+getattr(analyzer,'last_front_trim_area',0))
        results.append(result);profiles.append(profile);observed.append(confirmed);possible.append(potential)
        save_image(output/'masks'/f'{number:06d}.png',main.astype(np.uint8)*255)
        if (i+1)%50==0:print(name,'measure',i+1,'/',len(files),flush=True)
    impact=detect_impact(results,settings,metadata.get('frame_rate'))
    wall=analyzer.wall
    analysis_window=limit_valid_view(results,wall,analyzer.roi)
    signature=hashlib.sha256((signature+json.dumps(sources,sort_keys=True)).encode()).hexdigest()[:20]
    manifest=dict(schema_version=5,algorithm_version=VERSION,algorithm_sha256=source_digest,sequence=name,analysis_signature=signature,
        source_directory=str(directory),frame_numbering='Original numeric filename suffix; never renumbered.',
        coordinate_system='reference-aligned image pixels; origin top-left, x right, y down',units='px',tool_version='0.1.0',
        physical_calibration=dict(status='not_configured',mm_per_px=None),
        image_size=[analyzer.background.shape[1],analyzer.background.shape[0]],settings=asdict(settings),camera_metadata=metadata,alignment_cache=cache_provenance,
        wall=dict(source=wall.source,points=wall.points.tolist(),normals=wall.normals.tolist(),s_px=wall.s.tolist(),uncertainty_px=settings.wall_uncertainty_px),
        impact=impact,analysis_window=analysis_window,definitions=dict(contact_length_px='Wall arc between outermost main-film contact endpoints, including the span of internal uncertain gaps. Not sum of evidence intervals.',
            contact_evidence_length_px='Sum of separately image-supported main-film contact intervals.',
            maximum_visible_thickness='Maximum solid-safe wall-normal distance to the complete main-liquid outer silhouette. Bright internal highlights are included; independent drops excluded.',
            contours='Complete main film only, including front body and attached elongated tail. Other detected drops are in satellite_contours.',
            uncertainty='Image-evidence flags, not calibrated statistical confidence intervals.'),frames=results)
    attach_impact_point(manifest)
    dump(output/'measurements.json',manifest);dump(output/'source_frames.json',sources)
    pixel_csv(output/'measurements.csv',manifest);pixel_csv(output/'postimpact_measurements.csv',manifest);pixel_csv(output/'all_frames_trace.csv',manifest,True)
    np.savez_compressed(output/'profiles.npz',frame_numbers=numbers,s_px=wall.s,wall_points=wall.points,normals=wall.normals,
        thickness_px=profiles,supported_contact=observed,possible_contact=possible)
    selected=set(numbers[k] for k in np.linspace(0,len(numbers)-1,8,dtype=int))
    impact_number=impact['first_contact_frame'];near=[n for n in numbers if impact_number is not None and abs(n-impact_number)<=3]
    selected.update(near);by_number={n:i for i,n in enumerate(numbers)}
    def make_rows(ns):return [(n,base(by_number[n])[0],wall,results[by_number[n]]) for n in ns]
    sheet(make_rows(sorted(selected)),output/'review_sheet.png',name+' | V5 main film')
    sheet(make_rows(near or numbers[:3]),output/'impact_review.png',name+' | V5 contact onset')
    for n in sorted(selected):save_image(output/'overlays'/f'{n:06d}.png',overlay(base(by_number[n])[0],wall,results[by_number[n]]))
    series_plots(output,manifest);write_review(output/'review.html',manifest)
    summary=dict(sequence=name,frames_processed=len(results),first_contact=impact,analysis_window=analysis_window,postimpact_valid_frames=sum(r['phase']=='postimpact' for r in results),flag_counts=dict(Counter(f for r in results for f in r['flags'])))
    dump(output/'summary.json',summary);print(name,': completed; impact',impact_number,flush=True)
    return summary
