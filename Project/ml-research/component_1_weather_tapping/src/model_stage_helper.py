from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'outputs/reports'; REPORTS.mkdir(parents=True,exist_ok=True)
def stage_report(stage,title,artifact_glob,output):
    files=sorted(ROOT.glob(artifact_glob))
    rows=[]
    for p in files:
        if p.is_file(): rows.append({'stage':stage,'stage_title':title,'artifact':str(p.relative_to(ROOT)),'bytes':p.stat().st_size})
    pd.DataFrame(rows,columns=['stage','stage_title','artifact','bytes']).to_csv(REPORTS/output,index=False)
    print(f'{title}: found {len(rows)} artifacts; report: {REPORTS/output}')
