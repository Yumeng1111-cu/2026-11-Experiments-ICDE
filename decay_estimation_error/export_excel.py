#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Monte Carlo summaries to Excel (separate from simulation/plotting).
Uses artifact_tool for workbook creation, standard-library CSV for reading.
Full raw simulation records remain in CSV to keep Excel responsive.
"""
import argparse
import csv
import json
from pathlib import Path
from artifact_tool import Workbook, SpreadsheetFile

ROOT=Path(__file__).resolve().parent
COLORS={'ACRP-FV':'#C84C63','SW-FV':'#E6B84F','SW-UCB-FV':'#3977AE','Fixed-Share':'#956943','Calibrated Full-history':'#2C9D80','Oracle-SW':'#765AA5'}
METHODS=['ACRP-FV','SW-FV','SW-UCB-FV','Fixed-Share','Calibrated Full-history','Oracle-SW']
FILES=[('RRL stats','summary/summary_rrl.csv'),('ACRP paired CI','summary/ACRP_paired_vs_correct_mean.csv'),('Windows','summary/window_summary.csv'),('Oracle windows','summary/oracle_window.csv'),('Selected parameters','raw/selected_parameters.csv')]

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        rows=list(rd)
        return list(rd.fieldnames),rows

def value(x):
    if x is None or x=='':return None
    try:
        f=float(x)
        if f.is_integer() and len(str(x))<14:return int(f)
        return f
    except (ValueError,OverflowError):return x

def put_sheet(wb,name,headers,rows,first_col=20):
    ws=wb.worksheets.add(name)
    vals=[headers]+[[value(x) for x in r] for r in rows]
    end=len(vals)
    ws.get_range_by_indexes(0,0,end,len(headers)).values=vals
    header=ws.get_range_by_indexes(0,0,1,len(headers))
    header.format={'fill':'#233648','font':{'bold':True,'color':'#FFFFFF'},'row_height':25,'horizontal_alignment':'center'}
    ws.freeze_panes.freeze_rows(1)
    ws.get_range_by_indexes(0,0,end,len(headers)).format.column_width=19
    ws.get_range_by_indexes(0,0,end,1).format.column_width=first_col
    if end>1 and len(headers)>1:
        ws.get_range_by_indexes(1,1,end-1,len(headers)-1).set_number_format('0.0000')
    return ws

def main(root):
    wb=Workbook.create()
    headers,rrl=read_csv(root/'summary/summary_rrl.csv')
    lookup={(float(row['relative_error']),row['method']):float(row['mean_rrl_pct']) for row in rrl}
    errors=sorted({float(r['relative_error']) for r in rrl})
    mat=[]
    for e in errors:
        mat.append([e*100,0.042*(1+e)]+[lookup[(e,m)] for m in METHODS])
    ws=put_sheet(wb,'RRL comparison',['Error (%)','Assumed mean']+METHODS,mat,first_col=15)
    ws.get_range('C2:C10').format.font.color=COLORS['ACRP-FV']
    ws.get_range('C2:C10').format.font.bold=True
    ws.get_range('A1:H1').format.row_height=32
    for name,src in FILES:
        h,data=read_csv(root/src)
        put_sheet(wb,name,h,[[r[k] for k in h] for r in data])
    cfg=json.loads((root/'experiment_config.json').read_text(encoding='utf8'))
    readme=put_sheet(wb,'Readme',['Field','Value'],[
        ['Mean time sensitivity (true)',cfg['true_mean_theta']],['CV',cfg['cv']],
        ['Training MC / error',str(cfg['oracle_mc'])+' (independent Oracle-SW)'],
        ['Test MC / error',str(cfg['mc'])+' shared test histories across errors'],
        ['RRL unit','percent; lower is better'],
        ['Raw all-method records','See raw/all_method_runs.csv; all rows preserved'],
        ['Full parameter records','See Selected parameters + raw/selected_parameters.csv'],
        ['Price grid','81 geometric prices over [0.01,5]'],
        ['Word or Figure 4','Not generated'],
    ],first_col=36)
    path=root/'summary/mean_decay_error_500MC.xlsx'
    SpreadsheetFile.export_xlsx(wb).save(str(path))
    print('EXCEL saved:',path)
    try:
        print(wb.inspect({'kind':'table','range':'RRL comparison!A1:H4','include':'values','table_max_rows':4,'table_max_cols':8}).ndjson[:1800])
    except Exception as exc:
        print('Inspection skip:',str(exc)[:150])
    return path

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',default='runs/validation_cv04_500mc')
    args=parser.parse_args()
    root=Path(args.data_root)
    if not root.is_absolute():root=ROOT/root
    main(root)
