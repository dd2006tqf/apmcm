# figures/_figcommon.py — shared Nature-style boilerplate for the gen_fig_*.py scripts
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use('Agg')
from _utils.plot_utils import (setup_style, save_fig, nature_palette, nature_markers,
                               set_paper_placement, auto_legend, shared_legend,
                               consolidate_shared_legends, uncertainty_band, dynamic_limits,
                               declutter_axes, draw_vector_heatmap, smart_labels, PALETTE, COLORS)
setup_style(palette='nature')
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(ROOT, 'figures')
OUT_DIR = os.path.join(ROOT, 'output')
CACHE_Q1 = os.path.join(ROOT, '_tmp', 'cache', 'q1')
PUB = os.path.join(ROOT, 'user_data', 'a题数据汇总', '公开数据')
C = nature_palette()
M = nature_markers()

# pinyin names for the Hengzhou places that appear in figure labels
PINYIN = {
    '横州市': 'Hengzhou City', '横州镇': 'Hengzhou Town', '峦城镇': 'Luancheng', '校椅镇': 'Xiaoyi',
    '云表镇': 'Yunbiao', '陶圩镇': 'Taoxu', '马山镇': 'Mashan', '六景镇': 'Liujing', '那阳镇': 'Nayang',
    '平朗镇': 'Pinglang', '平马镇': 'Pingma', '莲塘镇': 'Liantang', '石塘镇': 'Shitang', '百合镇': 'Baihe',
    '镇龙乡': 'Zhenlong', '马岭镇': 'Maling', '南乡镇': 'Nanxiang', '高村小学': 'Gaocun Sch.',
    '横州市高级中学': 'Hengzhou High Sch.', '横州市第二初级中学': 'No.2 Middle Sch.', 'school_1': 'School 1',
    '旺宅': 'Wangzhai', '新兴': 'Xinxing', '杨江': 'Yangjiang', '新张': 'Xinzhang', '双河': 'Shuanghe',
    '飘竹': 'Piaozhu', '泮塘': 'Pantang', '潭宽': 'Tankuan', '新塘': 'Xintang', '覃寨': 'Qinzhai',
    '石板': 'Shiban', '六景工业园区': 'Liujing Ind. Park', '福塘': 'Futang', '里造': 'Lizao', '石村': 'Shicun',
    '新成': 'Xincheng', '苏村': 'Sucun', '逢村': 'Fengcun',
}


def py(name):
    return PINYIN.get(name, name)


def load(name):
    """Read a workspace JSON from figures/ or output/; missing data must fail loudly."""
    for d in (FIG_DIR, OUT_DIR):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(name)


def panel(ax, letter):
    ax.set_title(letter, loc='left', fontweight='bold', pad=3)


def finish(fig, name):
    save_fig(fig, os.path.join(FIG_DIR, name + '.pdf'))
