"""Render actual training and paired held-out results as PNG/SVG figures."""
import argparse
import json
import math
import os
from pathlib import Path

from .deepseek_trajectories import atomic_json
from .prepare_sft import sha256
from .report_teacher_eval import GROUPS


def cumulative(values):
    total, result = 0, []
    for i, v in enumerate(values, 1):
        total += v
        result.append(100*total/i)
    return result


def cumulative_likelihood(rows):
    nll, tokens, ce, ppl = 0., 0, [], []
    for row in rows:
        nll += row['nll_sum']
        tokens += row['target_tokens']
        ce.append(nll/tokens)
        ppl.append(math.exp(nll/tokens))
    return ce, ppl


def plot(comparison, training, likelihood):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent.parent/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter, ScalarFormatter
    import numpy as np

    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'svg.fonttype': 'none',
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'axes.spines.top': False,
        'axes.spines.right': False, 'font.size': 10, 'axes.titleweight': 'bold',
        'axes.labelcolor': '#334155', 'text.color': '#17243a', 'axes.edgecolor': '#cbd5e1',
        'xtick.color': '#475569', 'ytick.color': '#475569'})
    result = json.loads((comparison/'metrics.json').read_text(encoding='utf-8'))
    train = json.loads((training/'report.json').read_text(encoding='utf-8'))
    count = len(result['cases'])
    split_label = '验证' if result.get('split') == 'validation' else '测试'
    assert result['test_cases'] == count and count > 0
    assert train['completed_updates'] == len(train['steps'])
    source_rows = {n: json.loads((likelihood/f'{n}-actions.json').read_text(encoding='utf-8')) for n in ('base', 'sft')}
    reference_count = len(source_rows['base'])
    assert reference_count == len(source_rows['sft']) and reference_count > 0
    assert [(r['case_id'],r['turn']) for r in source_rows['base']] == [(r['case_id'],r['turn']) for r in source_rows['sft']]
    loss = [r['loss'] for r in train['steps']]
    assert all(v >= 0 and math.isfinite(v) for v in loss)
    smooth = [sum(loss[max(0,i-19):i+1])/len(loss[max(0,i-19):i+1]) for i in range(len(loss))]
    curves = {'training': {'updates': list(range(1,len(loss)+1)), 'raw_loss': loss, 'rolling_20_loss': smooth},
        'test': {n: cumulative([r[f'{n}_pass'] for r in result['cases']]) for n in ('base','sft')},
        'reference_likelihood': {n: dict(zip(('cross_entropy','perplexity'),cumulative_likelihood(source_rows[n]))) for n in ('base','sft')}}
    out = comparison/'figures'
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'chart-data.json', curves)
    colors = {'base': '#c46c36', 'sft': '#1666bf'}
    labels = {'base': '训练前：原始基座', 'sft': '训练后：SFT'}
    def decorate(ax):
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    def training_curve(ax):
        ax.plot(curves['training']['updates'], loss, color=colors['sft'], alpha=.22, linewidth=.7, label='每步原始 loss')
        ax.plot(curves['training']['updates'], smooth, color=colors['sft'], linewidth=1.6, label='20 步移动平均')
        ax.set(xlabel='优化器更新步数', ylabel='训练 loss', title=f'A  训练 loss：{len(loss)} 次真实更新')
        if all(v > 0 for v in loss):
            ax.set_yscale('log'); ax.set_ylabel('训练 loss（对数轴）')
        ax.legend(frameon=False, fontsize=9)
        decorate(ax)
    def success_curve(ax):
        for n in ('base','sft'):
            y = curves['test'][n]
            ax.plot(range(1,count+1), y, color=colors[n], label=f'{labels[n]}  {y[-1]:.2f}%', linewidth=1.8)
            ax.scatter([count], [y[-1]], color=colors[n], s=20)
        ax.set(xlabel=f'已评测题数（固定{split_label}集顺序）', ylabel='累计完整任务成功率', ylim=(-4,105), title='B  训练前后：完整任务成功率')
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.legend(frameon=False, fontsize=9, loc='center right')
        decorate(ax)
    def ppl_curve(ax):
        for n in ('base','sft'):
            y = curves['reference_likelihood'][n]['perplexity']
            ax.plot(range(1,reference_count+1), y, color=colors[n], label=f'{labels[n]}  {y[-1]:.4f}', linewidth=1.8)
        ax.set(xlabel=f'累计{split_label}参考动作数', ylabel='目标 token 加权 PPL', yscale='log', title=f'C  {split_label}参考动作：累计困惑度')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.legend(frameon=False, fontsize=9)
        decorate(ax)
    def ce_curve(ax):
        for n in ('base','sft'):
            y = curves['reference_likelihood'][n]['cross_entropy']
            ax.plot(range(1,reference_count+1),y,color=colors[n],label=f'{labels[n]}  {y[-1]:.6f}',linewidth=1.8)
        ax.set(xlabel=f'累计{split_label}参考动作数',ylabel='目标 token 加权交叉熵（nats/token，对数轴）',
               yscale='log',title=f'训练前后：{split_label}参考动作累计交叉熵')
        ax.legend(frameon=False,fontsize=9)
        decorate(ax)
    fields = [('strict_pass_at_1', '完整任务成功率'), ('answer_exact_match','答案 EM'),
              ('status_macro_f1','状态 Macro F1'), ('numeric_exact_with_unit','数值/单位正确'),
              ('valid_model_actions','动作协议合法'), ('token_accuracy','参考 Token Accuracy')]
    def field_value(n, key):
        if key == 'token_accuracy':
            return 100*result['likelihood'][n]['teacher_forced_token_accuracy']
        value = result['metrics'][n][key]
        return 100*(value['rate'] if isinstance(value,dict) else value)
    def metrics_chart(ax):
        y = np.arange(len(fields))
        for n, delta in [('base',-.18),('sft',.18)]:
            v = [field_value(n,k) for k,_ in fields]
            ax.barh(y+delta,v,height=.32,color=colors[n],label=labels[n])
            for yi, vi in zip(y+delta,v):
                ax.text(vi+1,yi,f'{vi:.2f}%',va='center',fontsize=8,color=colors[n])
        ax.set(yticks=y,yticklabels=[v for _,v in fields],xlim=(0,117),title='D  训练前后：主要指标')
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.grid(axis='x',alpha=.2)
        ax.set_axisbelow(True)
    def save(fig, name):
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
        fig.savefig(out/f'{name}.svg',bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(16,10.5))
    training_curve(axes[0,0]); success_curve(axes[0,1]); ppl_curve(axes[1,0]); metrics_chart(axes[1,1])
    fig.suptitle('Qwen3-1.7B 财报工具任务：训练前后对比',fontsize=19,fontweight='bold',y=.98)
    fig.text(.5,.017,f'{count} 道按公司隔离的评测题；{reference_count} 条参考动作只用于 Loss/PPL。B、C 横轴是评测样本数，不是训练步数。',ha='center',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.07,right=.96,top=.90,bottom=.08,wspace=.34,hspace=.35)
    save(fig,'overview')
    for name, renderer in [('training-loss',training_curve),('test-success',success_curve),('test-perplexity',ppl_curve),('test-cross-entropy',ce_curve),('metrics-before-after',metrics_chart)]:
        fig, ax = plt.subplots(figsize=(10,5.4))
        renderer(ax)
        if name == 'metrics-before-after':
            ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2,fontsize=9)
        fig.tight_layout()
        save(fig,name)
    fig, ax = plt.subplots(figsize=(12,6.8))
    groups = sorted(result['metrics']['sft']['by_group'])
    y = np.arange(len(groups))
    for n, delta in [('base',-.18),('sft',.18)]:
        values = [100*result['metrics'][n]['by_group'][g]['rate'] for g in groups]
        ax.barh(y+delta,values,height=.32,color=colors[n],label=labels[n])
        for yi, v, g in zip(y+delta,values,groups):
            item = result['metrics'][n]['by_group'][g]
            ax.text(v+1,yi,f"{item['correct']}/{item['total']}  {v:.1f}%",va='center',fontsize=9,color=colors[n])
    ax.set(yticks=y,yticklabels=[GROUPS.get(g,g) for g in groups],xlim=(0,124),title=f'{len(groups)} 类题型：训练前后完整任务成功率')
    ax.invert_yaxis(); ax.xaxis.set_major_formatter(PercentFormatter()); ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.08),ncol=2); fig.tight_layout(); save(fig,'by-group')
    atomic_json(out/'manifest.json', {'matplotlib_version': matplotlib.__version__, 'script_sha256': sha256(Path(__file__)),
        'comparison_metrics_sha256': sha256(comparison/'metrics.json'),
        'files_sha256': {p.name:sha256(p) for p in sorted(out.glob('*')) if p.is_file() and p.name!='manifest.json'},
        'note': 'Only training loss is indexed by optimizer updates. Test curves accumulate held-out samples, not model checkpoints.'})
    readme = comparison/'README.md'
    marker = '\n## 前后对比与曲线图\n'
    text = readme.read_text(encoding='utf-8').split(marker)[0]
    readme.write_text(text+marker+'\n![训练前后对比](figures/overview.png)\n\n'
        '逐题成功率和累计 PPL 曲线按评测样本数绘制，并非每个训练 checkpoint 的验证曲线。'
        '目前只评测了原始基座与最终 SFT，不虚构中间 checkpoint 的测试成绩。\n\n'
        '[分题型对比](figures/by-group.png) · [交叉熵曲线](figures/test-cross-entropy.png) · [曲线原始数据](figures/chart-data.json) · '
        '[可编辑 SVG 总图](figures/overview.svg)\n',encoding='utf-8')
    print(str(out/'overview.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--likelihood',type=Path,required=True)
    args=parser.parse_args()
    plot(args.comparison.resolve(),args.training.resolve(),args.likelihood.resolve())
