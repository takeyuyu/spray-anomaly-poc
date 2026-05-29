"""
噴霧（スプレー）異常検知 最小PoC
=====================================
「噴霧の形」ではなく「明るさの時間変化」を見ることで、
正常 / チラチラ（予兆）/ 詰まり の3つを見分けられるかを、合成データで実証する。

守秘配慮: 実機データは一切使用していない。すべてプログラムで生成した合成データ。

考え方:
  カメラ映像が暗くなる/乱れる原因を、明るさが時間でどう動くかで捉える。
  - 正常        : 明るさが高いまま安定
  - チラチラ(予兆): 明るさが速く上下にバタつく
  - 詰まり        : 明るさがゼロに張り付く
  完全に詰まる前の「チラチラ」段階で気づければ、手遅れではなく予知になる。

実行すると、同じフォルダに以下が出力される:
  - poc_frames.png : 窓越し・ダストを模した合成フレームの例（静止画では見分けにくい）
  - poc_result.png : 明るさの時間変化で3状態が分離する様子
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

# 日本語フォント（環境になければ自動で代替フォントになる）
try:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    BOLD = fm.FontProperties(family="Noto Sans CJK JP", weight="bold")
    JP = fm.FontProperties(family="Noto Sans CJK JP")
except Exception:
    BOLD = None; JP = None
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---- 撮影条件（窓越し・ダストを模す）----
FPS = 30
SECONDS = 10
N = FPS * SECONDS
H = W = 64

# ---- 噴霧コーンの生成 ----
def spray_cone(intensity):
    yy, xx = np.mgrid[0:H, 0:W]
    cx = W/2
    img = np.zeros((H, W))
    if intensity > 0:
        for y in range(H):
            width = 2 + (y/H)*(W*0.35)
            img[y] = np.exp(-((xx[y]-cx)**2)/(2*width**2)) * (y/H)**0.4
        img = img/img.max()*intensity
    return img

def make_background(n):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W/2, H/2
    haze = 0.18*np.exp(-(((xx-cx)**2+(yy-cy)**2)/(2*(W*0.6)**2)))
    drift = 0.05*np.sin(np.linspace(0,2*np.pi,n))[:,None,None]
    return haze[None,:,:] + drift

def add_dust(stack):
    n = stack.shape[0]
    out = stack + make_background(n)
    dust = (rng.random(out.shape)>0.985).astype(float)*rng.uniform(0.1,0.3,out.shape)
    out = out + dust + rng.normal(0,0.04,out.shape)
    return np.clip(out,0,1)

# ---- 3つの状態の「噴霧の強さ」の時間プロファイル ----
def prof_normal(n):
    return np.clip(0.85+0.03*np.sin(np.linspace(0,6*np.pi,n))+rng.normal(0,0.02,n),0,1)

def prof_flicker(n):
    base=0.8*np.ones(n); t=np.arange(n)
    flick=(np.sin(2*np.pi*t/4)>-0.2).astype(float)
    drop=(rng.random(n)>0.5); sig=base*flick; sig[drop&(flick<1)]=0
    return np.clip(sig+rng.normal(0,0.03,n),0,1)

def prof_clog(n):
    sig=prof_flicker(n); onset=int(n*0.45); sig[onset:]=0
    for p in [int(n*0.6),int(n*0.78)]: sig[p:p+2]=0.5
    return np.clip(sig,0,1)

conds = {"正常":prof_normal(N), "チラチラ（予兆）":prof_flicker(N), "詰まり":prof_clog(N)}

# ---- 各状態の動画から「明るさの時系列」を抽出 ----
def brightness(prof):
    frames = add_dust(np.stack([spray_cone(i) for i in prof], axis=0))
    region = frames[:,:,W//2-14:W//2+14][:,4:H-4,:]
    return region.mean(axis=(1,2))

def hf(b):
    win = max(3, FPS//6)   # 約0.16秒窓での短時間のバタつき
    return np.array([b[max(0,i-win):i+win].std() for i in range(len(b))])

res = {}
for name, prof in conds.items():
    b = brightness(prof)
    res[name] = {"b":b, "hf":hf(b), "hf_mean":hf(b).mean(),
                 "zero":float((b<0.15).mean())}

# ---- 判定（正常の冒頭2秒を基準に閾値を決める）----
base_n = FPS*2
TH_FLICK = res["正常"]["hf"][:base_n].mean() + 5*res["正常"]["hf"][:base_n].std()

print("="*56)
print("噴霧 異常検知 PoC 判定結果")
print("="*56)
for name, r in res.items():
    if r["zero"] > 0.25:
        verdict = "詰まり"
    elif r["hf_mean"] > TH_FLICK:
        verdict = "チラチラ（予兆）"
    else:
        verdict = "正常"
    print(f"  {name:<12} → 判定: {verdict}")
print("="*56)

colors = {"正常":"#0f6e63", "チラチラ（予兆）":"#c2641f", "詰まり":"#a32a2a"}
t = np.arange(N)/FPS

# ============ 図1: 合成フレーム（静止画では見分けにくい）============
fig = plt.figure(figsize=(11, 5.2)); fig.patch.set_facecolor("white")
fig.text(0.5,0.96,"【ポイント】下の3枚は実は別々の状態。でも…静止画だと見分けがつきません",
         ha="center",va="top",fontsize=14,color="#a32a2a",fontproperties=BOLD)
fig.text(0.5,0.90,"窓が曇り、ダストが舞う現場のカメラ映像（を模した合成画像）",
         ha="center",va="top",fontsize=10.5,color="#41506a",fontproperties=JP)
frame_labels = {"正常":"正常",
                "チラチラ（予兆）":"チラチラ\n(この瞬間は噴いている)",
                "詰まり":"詰まり\n(この直前まで噴いていた)"}
# どれも「噴いている瞬間」を撮る（だから同じに見える）
for i,(name,disp) in enumerate(frame_labels.items()):
    ax=fig.add_subplot(1,3,i+1)
    one = add_dust(spray_cone(0.80)[None,:,:])[0]
    ax.imshow(one, cmap="gray", vmin=0, vmax=1)
    ax.set_title(disp, fontsize=12, fontproperties=BOLD, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor("#c9c0ad"); s.set_linewidth(1.5)
fig.text(0.5,0.06,"→ だから「形」で見分けるのをあきらめ、「明るさが時間でどう変わるか」を見ることにした",
         ha="center",va="bottom",fontsize=12,color="#0f6e63",fontproperties=BOLD)
plt.subplots_adjust(top=0.82, bottom=0.14, wspace=0.1)
plt.savefig("poc_frames.png", dpi=140, bbox_inches="tight", facecolor="white")
print("保存: poc_frames.png")

# ============ 図2: 明るさの時間変化で分離 ============
fig = plt.figure(figsize=(12, 8.5)); fig.patch.set_facecolor("white")
gs = GridSpec(3,3,figure=fig,hspace=0.55,wspace=0.35,height_ratios=[1,1,1.25])
fig.text(0.5,0.975,"時間で見ると、3つの状態がはっきり分かれる",
         ha="center",va="top",fontsize=15,color="#0f6e63",fontproperties=BOLD)
fig.text(0.5,0.94,"静止画では見分けられなかった3つを、「明るさの時間変化」で捉える",
         ha="center",va="top",fontsize=10.5,color="#41506a",fontproperties=JP)
for j,(name,r) in enumerate(res.items()):
    ax=fig.add_subplot(gs[0,j])
    ax.plot(t,r["b"],color=colors[name],lw=0.9)
    ax.set_title(name+" の明るさ",fontsize=11,fontproperties=BOLD)
    ax.set_ylim(-0.05,1.05); ax.set_xlabel("時間（秒）",fontsize=8.5); ax.tick_params(labelsize=7)
    if j==0: ax.set_ylabel("明るさ",fontsize=8.5,fontproperties=JP)
notes=["ずっと高く安定","激しく上下にバタつく","途中からゼロに張り付く"]
for j,note in enumerate(notes):
    ax=fig.add_subplot(gs[1,j]); ax.axis("off")
    ax.text(0.5,0.85,"↑ 見え方",ha="center",fontsize=9,color="#888",fontproperties=JP)
    ax.text(0.5,0.5,note,ha="center",va="center",fontsize=13,
            color=list(colors.values())[j],fontproperties=BOLD)
ax=fig.add_subplot(gs[2,:2])
for name,r in res.items():
    ax.scatter(r["hf_mean"],r["zero"],s=320,color=colors[name],
               edgecolor="white",linewidth=2.5,zorder=3)
    ax.annotate(name,(r["hf_mean"],r["zero"]),fontsize=11,xytext=(12,8),
                textcoords="offset points",color=colors[name],fontproperties=BOLD)
ax.set_xlabel("バタつきの大きさ →（チラチラほど右）",fontsize=10,fontproperties=BOLD)
ax.set_ylabel("止まっている割合 →（詰まりほど上）",fontsize=10,fontproperties=BOLD)
ax.set_title("3つの状態は、別々の場所にきれいに分かれた",fontsize=12,fontproperties=BOLD)
ax.tick_params(labelsize=8.5); ax.grid(True,alpha=0.2)
ax=fig.add_subplot(gs[2,2]); ax.axis("off")
summary=("【結果】\n\n「明るさが時間で\n どう動くか」だけで、\n\n"
         "・正常\n・チラチラ（予兆）\n・詰まり\n\n の3つを\n 見分けられた。\n\n"
         "しかも、詰まる前の\n チラチラ段階で\n 気づける＝予知。")
ax.text(0.05,0.95,summary,fontsize=10.5,va="top",color="#1a2230",linespacing=1.5,fontproperties=JP)
plt.subplots_adjust(top=0.90, bottom=0.06)
plt.savefig("poc_result.png", dpi=140, bbox_inches="tight", facecolor="white")
print("保存: poc_result.png")
