"""Human-readable provenance and ordering, independent of numerical identity."""
from pathlib import Path
import re


def natural_key(value):
    return tuple((1, float(part)) if re.fullmatch(r'\d+(?:\.\d+)?', part) else (0, part.casefold())
                 for part in re.split(r'(\d+(?:\.\d+)?)', str(value)))


def source_info(directory, input_root=None):
    directory=Path(directory)
    parts=directory.parts
    conditions=[p for p in parts if re.search(r'\d+(?:\.\d+)?\s*mm[_ -]+\d+(?:\.\d+)?\s*mm',p,re.I)]
    angles=[p for p in parts if re.fullmatch(r'\d+(?:\.\d+)?\s*[°度]',p)]
    views=[p for p in parts if p in ['侧面','侧视','正面','正视','俯视','底视','side','front','top']]
    context=[*angles[-1:],*views[-1:],*conditions[-1:]]
    try:relative=directory.relative_to(Path(input_root)).as_posix() if input_root else directory.name
    except ValueError:relative=directory.name
    if not context:
        context=[relative if relative!='.' else directory.name]
        if re.match(r'BMP\d',directory.name,re.I):context=[directory.parent.name,directory.name]
    label=' / '.join(context)
    if conditions and directory.name!=conditions[-1]:
        # Camera acquisition IDs disambiguate repeat captures of one condition.
        acquisition=directory.name
    else:acquisition=''
    return dict(display_label=label,source_directory=str(directory),relative_directory=relative,
                angle=angles[-1] if angles else None,view=views[-1] if views else None,
                condition=conditions[-1] if conditions else None,acquisition=acquisition)


def readable_name(label,limit=48):
    value=re.sub(r'[<>:"/\\|?*#%\x00-\x1f]+','__',label)
    value=re.sub(r'\s+','_',value).strip(' ._')
    value=re.sub(r'_{2,}','__',value)
    value=value[:limit].rstrip(' ._') or 'dataset'
    if re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])',value,re.I):value='data_'+value
    return value


def entry_folder(entry):
    return entry.get('output_directory',entry['id'])


def display_manifest(manifest):
    # Never change sequence/signature: old reviews must still match exactly.
    value=dict(manifest)
    info=source_info(value.get('source_directory',value.get('sequence','dataset')))
    value.setdefault('source_info',info)
    value.setdefault('display_label',info['display_label'])
    return value


def dataset_key(item):
    info=item.get('source_info') or source_info(item.get('directory',item.get('source_directory',item.get('sequence',''))))
    return (natural_key(info['display_label']),natural_key(info.get('relative_directory','')),
            natural_key(info.get('acquisition','')),str(item.get('analysis_signature',item.get('id',''))))
