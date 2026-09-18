"""只读诊断 PB 覆盖率；保留旧口径，协议修正另列，不覆盖冻结结果。"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

DT, TAU, ALPHA, MU, AMAX, D = .05, .70, .5, 2., 2., .8


def affine(state):
    """返回旧脚本同义约束及原始径向量；非激活行仍保留零裕度。"""
    p, p2, v, v2 = state
    r, w = p[:2]-p2[:2], v[:2]-v2[:2]
    d = float(np.linalg.norm(r))
    if d <= 1e-9:
        raise ValueError('重合状态超出原审计可微域，禁止静默删除')
    n = r/d
    radial = float(n@w)
    h = d-D-min(radial, 0.)**2/(2*MU)
    drift = radial+ALPHA*h
    if radial < 0:
        lam = -radial/MU
        c = np.r_[lam*n, -lam*n]
        b = -ALPHA*h-radial-lam*(float(w@w)-radial**2)/d
        branch = '接近且约束激活'
    else:
        c, b = np.zeros(4), 0. if drift >= 0 else 1.
        branch = '非接近且约束省略' if drift >= 0 else '非接近但不可行'
    return c, b, branch, radial, drift


def physical_value(q, x):
    """不将省略约束替换为零的解析屏障表达式；仅用于解释编码跳变。"""
    c,b,branch,_,drift=affine(q)
    return float(c@x-b) if branch=='接近且约束激活' else drift


def state(row):
    d = row['drones']
    return tuple(np.asarray(d[str(i)][key], float) for key, i in
                 [('pos',2),('pos',3),('v_actual',2),('v_actual',3)])


def action(row):
    x = np.r_[row['drones']['2']['a_safe'], row['drones']['3']['a_safe']]
    if x.shape != (4,) or not np.isfinite(x).all():
        raise ValueError('动作维度或有限性错误')
    return x


def predicted(row):
    p,p2,v,v2 = state(row)
    u,u2 = (np.asarray(row['drones'][str(i)]['v_requested'],float)[:2] for i in (2,3))
    al = 1-np.exp(-DT/TAU)
    be = DT-TAU*al
    ans = [z.copy() for z in (p,p2,v,v2)]
    for idx, vel, req in [(0,v,u),(1,v2,u2)]:
        ans[idx][:2] += vel[:2]*DT+be*(req-vel[:2])
        ans[idx+2][:2] = (1-al)*vel[:2]+al*req
    return tuple(ans)


def residuals(now, nxt):
    x = action(now)
    def f(q):
        c,b,*_ = affine(q)
        return float(c@x-b)
    actual = f(state(nxt))
    return abs(actual-f(state(now))), abs(actual-f(predicted(now)))


def traces(root):
    paths = sorted(p for p in root.glob('*AEGIS_HOCBF_V4/*.jsonl') if not p.name.startswith('._'))
    if not paths:
        raise ValueError(f'没有轨迹，禁止生成空结果：{root}')
    return [(p,[json.loads(l) for l in p.read_text().splitlines()]) for p in paths]


def summarize(rows):
    out = {'样本数':len(rows)}
    if not rows:
        return out
    for key in ['eta','selected','optimality_gap','legacy_residual','protocol_residual']:
        vals=[r[key] for r in rows if r.get(key) is not None]
        if vals:
            out[key]={'有效数':len(vals),'最小':min(vals),'中位':float(np.median(vals)),
                      'P95':float(np.quantile(vals,.95)),'最大':max(vals)}
    for key in ['legacy_eta_pass','legacy_selected_pass','protocol_eta_pass','protocol_selected_pass']:
        out[key]={'通过':sum(r[key] for r in rows),'分母':len(rows)}
    out['选中裕度不超过1e-6']=sum(r['selected']<=1e-6 for r in rows)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--archived-audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise SystemExit('输出已存在，拒绝覆盖')
    old=json.loads(args.archived_audit.read_text())
    calibration=traces(args.calibration)
    if len(calibration)!=5:
        raise ValueError('独立标定必须为五条轨迹')
    cal=[]
    for path,trace in calibration:
        for idx,(now,nxt) in enumerate(zip(trace,trace[1:])):
            if 'v_requested' not in now['drones']['2']:
                continue  # 精确保留旧标定筛选规则。
            legacy,protocol=residuals(now,nxt)
            x=action(now); actual=state(nxt); pred=predicted(now)
            cal.append({'轨迹':str(path),'行号':idx+1,'step':now.get('step'),
                'recovery_active':bool(now['drones']['2'].get('recovery_active')),
                '当前分支':affine(state(now))[2],'预测分支':affine(pred)[2],'实测分支':affine(actual)[2],
                'legacy_residual':legacy,'protocol_residual':protocol,
                'physical_prediction_residual':abs(physical_value(actual,x)-physical_value(pred,x))})
    old_delta=1.1*max(r['legacy_residual'] for r in cal)
    fixed_delta=1.1*max(r['protocol_residual'] for r in cal)
    if not np.isclose(old_delta,old['delta_B'],rtol=1e-9,atol=1e-9):
        raise ValueError('旧阈值无法复现，停止而不转用新阈值')
    allrows=[]
    inputs=[p for p,_ in calibration]+[args.archived_audit]
    per_trace=[]
    for entry in old['evaluation']:
        path=Path(entry['trajectory']);inputs.append(path)
        trace=[json.loads(l) for l in path.read_text().splitlines()]
        rows=[];age=0
        for idx,row in enumerate(trace):
            if not row['drones']['2']['recovery_active']:
                age=0;continue
            age+=1
            c,b,branch,radial,drift=affine(state(row));x=action(row)
            eta=float(AMAX*np.abs(c).sum()-b);selected=float(c@x-b)
            lr,pr=None,None
            if idx+1<len(trace) and 'v_requested' in row['drones']['2']:
                lr,pr=residuals(row,trace[idx+1])
            rows.append({'轨迹':str(path),'行号':idx+1,'接管连续步':age,
                '阶段':'前3步' if age<=3 else '后续步','分支':branch,
                'radial':radial,'nonclosing_physical_drift':drift,'eta':eta,'selected':selected,
                'optimality_gap':eta-selected,'legacy_residual':lr,'protocol_residual':pr,
                'legacy_eta_pass':eta>old_delta,'legacy_selected_pass':selected>old_delta,
                'protocol_eta_pass':eta>fixed_delta,'protocol_selected_pass':selected>fixed_delta})
        if len(rows)!=entry['recovery_steps'] or sum(r['legacy_eta_pass'] for r in rows)!=entry['covered_steps']:
            raise ValueError(f'旧逐轨迹统计未复现：{path}')
        allrows.extend(rows);per_trace.append({'轨迹':str(path),**summarize(rows)})
    if len(allrows)!=old['coverage']['steps'] or sum(r['legacy_eta_pass'] for r in allrows)!=old['coverage']['covered_steps']:
        raise ValueError('旧总计未复现')
    groups=defaultdict(list)
    for row in allrows:
        groups['分支/'+row['分支']].append(row)
        groups['阶段/'+row['阶段']].append(row)
        groups['场景/'+('C1' if 'postfreeze' not in row['轨迹'] else ('diagonal' if 'diagonal' in row['轨迹'] else ('jitter' if 'jitter' in row['轨迹'] else 'wide')))].append(row)
    report={'说明':'探索性机制审计，旧口径固定；协议修正由同五条标定轨迹计算，单列而不替代封存结果。实际选中输入是日志 a_safe，不是物理执行加速度。固定 D=0.8 延续旧审计，不是动态 clearance 的完整认证。',
        '旧经验界':old_delta,'按协议预测残差重新计算界':fixed_delta,
        '标定样本数':len(cal),
        '标定残差分位':{key:{'中位':float(np.median([r[key] for r in cal])),
            'P95':float(np.quantile([r[key] for r in cal],.95)),
            '最大':max(r[key] for r in cal)} for key in ['legacy_residual','protocol_residual','physical_prediction_residual']},
        '标定编码分支不一致数':sum(r['预测分支']!=r['实测分支'] for r in cal),
        '整体':summarize(allrows),'分组':{k:summarize(v) for k,v in groups.items()},'逐轨迹':per_trace,
        '旧界最大偏差位置':max(cal,key=lambda r:r['legacy_residual']),
        '协议残差最大位置':max(cal,key=lambda r:r['protocol_residual']),
        '输入sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    args.out.mkdir(parents=True)
    (args.out/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (args.out/'calibration_steps.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cal))
    (args.out/'steps.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
    print(json.dumps(report['整体'],ensure_ascii=False,indent=2))
if __name__=='__main__':
    main()
