"""Vector diagrams and scientific figures; every result curve comes from the solver."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Rectangle,Circle
ORANGE="#C86627"; BLUE="#205B89"; GRAY="#92999E"

def setup():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
      "axes.unicode_minus":False,"font.size":9,"axes.labelsize":9,"legend.fontsize":8,
      "xtick.labelsize":8,"ytick.labelsize":8,"axes.spines.top":False,"axes.spines.right":False,
      "axes.linewidth":.7,"grid.color":"#E5E7E9","grid.linewidth":.5,"mathtext.fontset":"stix",
      "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"path","savefig.facecolor":"white"})

def save(fig,folder,name):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for ext in ["pdf","svg","png"]:
        fig.savefig(folder/f"{name}.{ext}",dpi=300,bbox_inches="tight",pad_inches=.09)
    plt.close(fig)

def figure1(environment,folder):
    setup();fig,axes=plt.subplots(2,1,figsize=(6.3,4.8),sharex=True,layout="constrained")
    t=environment[:,0]/3600;fine=np.linspace(0,4,2001)
    for ax,col,color,label,panel in zip(axes,[1,2],[ORANGE,BLUE],
        [r"环境温度 $T_\infty$ / °C",r"等效水分驱动 $C_\infty^{\mathrm{eff}}$ / (kg/kg)"],["(a)","(b)"]):
        values=environment[:,col]-(273.15 if col==1 else 0)
        ax.axvspan(0,.5,color="#F4F1EB",zorder=0)
        ax.plot(fine,np.interp(fine,t,values),color=color,lw=1.3,zorder=3,label="分段线性插值")
        ax.scatter(t,values,s=4,color="#AAB1B7",alpha=.5,zorder=2,label="原始观测点")
        ax.set_ylabel(label);ax.set_xlim(0,4);ax.grid(True,alpha=.7)
        ax.text(.015,.93,panel,transform=ax.transAxes,va="top")
    axes[0].legend(loc="lower right",frameon=False,ncol=2)
    axes[0].text(.08,.78,"问题1\n结果范围",transform=axes[0].transAxes,ha="center",fontsize=8)
    axes[1].set_xlabel("时间 / h")
    save(fig,folder,"fig1_environment")

def figure2(folder):
    setup();fig,axes=plt.subplots(1,2,figsize=(6.3,3.9),layout="constrained",gridspec_kw={"width_ratios":[1,1.12]})
    ax=axes[0];ax.set_xlim(-.7,5.1);ax.set_ylim(-1.8,3.8);ax.axis("off")
    ax.add_patch(Rectangle((.5,.2),3,1.7,facecolor="#EAE2D6",edgecolor="none"))
    ax.add_patch(Ellipse((.5,1.05),.8,1.7,facecolor="#D9C9B1",edgecolor="#776C5F"))
    ax.add_patch(Ellipse((3.5,1.05),.8,1.7,facecolor="#F2ECE3",edgecolor="#776C5F"))
    ax.add_patch(Ellipse((2.2,1.05),.8,1.7,fill=False,edgecolor="#908577",ls=":",lw=.8))
    ax.plot([.5,3.5],[1.9,1.9],color="#776C5F");ax.plot([.5,3.5],[.2,.2],color="#776C5F")
    ax.annotate("",(.5,-.25),(3.5,-.25),arrowprops={"arrowstyle":"<->","color":"#454545"})
    ax.text(2,-.7,"L = 25 cm",ha="center")
    ax.annotate("",(3.5,1.05),(3.5,1.9),arrowprops={"arrowstyle":"<->","color":"#454545"})
    ax.text(3.8,1.45,"R = 2 cm",rotation=90,va="center")
    ax.text(.0,3.4,"(a) 圆柱整体（示意）")
    ax.text(2,2.65,"远离端部的代表性横截面",ha="center",fontsize=8)
    ax.annotate("",(2.2,1.93),(2.4,2.5),arrowprops={"arrowstyle":"->","color":GRAY})
    ax.text(2,-1.3,"一维径向近似，忽略轴向及端部差异",ha="center",fontsize=8)
    ax=axes[1];ax.set_xlim(-1.55,2.05);ax.set_ylim(-2.0,2.45);ax.set_aspect("equal");ax.axis("off")
    for r in [1,.8,.6,.35]:ax.add_patch(Circle((0,0),r,fill=False,edgecolor="#737E85",lw=.8,ls="-" if r==1 else ":"))
    ax.plot([0,1],[0,0],color="#343A40",lw=1.1);ax.scatter([0,1],[0,0],s=10,color="#343A40")
    ax.text(0,-.19,"r=0",ha="center",fontsize=8);ax.text(1.07,-.19,"r=R",fontsize=8)
    ax.text(.45,.13,"径向控制体",fontsize=8,ha="center")
    ax.annotate("",(.77,.7),(1.58,1.18),arrowprops={"arrowstyle":"->","color":ORANGE,"lw":1.5})
    ax.annotate("",(1.6,-.8),(.9,-.45),arrowprops={"arrowstyle":"->","color":BLUE,"lw":1.5})
    ax.text(.75,1.47,r"环境 $T_\infty(t),C_\infty^{\rm eff}(t)$",ha="center",fontsize=8)
    ax.text(1.1,.85,"热交换",color=ORANGE,fontsize=8);ax.text(1.06,-.9,"水分交换",color=BLUE,fontsize=8)
    ax.text(-1.3,2.15,"(b) 横截面与边界")
    ax.text(0,-1.3,r"中心：$T_r=C_r=0$",ha="center",fontsize=8)
    ax.text(.15,-1.65,r"表面：$-kT_r=h(T_s-T_\infty)$",ha="center",fontsize=8)
    ax.text(.15,-1.97,r"$-DC_r=h_m(C_s-C_\infty^{\rm eff})$",ha="center",fontsize=8)
    save(fig,folder,"fig2_radial_model")

def figure3(solution,folder):
    setup();fig,axes=plt.subplots(1,2,figsize=(6.3,3.5),layout="constrained")
    times=[100,300,600,900,1200,1500,1800]
    colors=plt.colormaps["Blues"](np.linspace(.42,.95,7));styles=["-","--","-.",":","-","--","-."]
    for t,color,style in zip(times,colors,styles):
        axes[0].plot(solution["radii"]*100,solution["temperature_K"][t]-273.15,color=color,ls=style,lw=1.3,label=f"{t} s")
        axes[1].plot(solution["radii"]*100,solution["moisture"][t],color=color,ls=style,lw=1.3)
    for ax,panel in zip(axes,["(a)","(b)"]):
        ax.set_xlabel("到中心的距离 / cm");ax.set_xlim(0,2);ax.set_xticks([0,.5,1,1.5,2]);ax.grid(True)
        ax.text(.02,1.02,panel,transform=ax.transAxes,va="bottom")
    axes[0].set_ylabel("温度 / °C");axes[1].set_ylabel("干基含水率 / (kg/kg)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=4,frameon=False)
    save(fig,folder,"fig3_q1_profiles")

def figure4(solution,environment,folder):
    setup();fig,axes=plt.subplots(2,2,figsize=(6.3,5.3),layout="constrained")
    time=solution["times"]/3600;r=solution["radii"]*100
    for j,key,cmap,label in [(0,"temperature_K","YlOrRd","温度 / °C"),(1,"moisture","Blues","干基含水率 / (kg/kg)")]:
        values=solution[key]-(273.15 if j==0 else 0)
        im=axes[0,j].pcolormesh(time,r,values.T,cmap=cmap,shading="auto",rasterized=True)
        bar=fig.colorbar(im,ax=axes[0,j],pad=.02,aspect=23);bar.set_label(label,fontsize=8)
        axes[0,j].set_ylabel("到中心的距离 / cm")
        for radius,color,style in [(0,"#243D56","-"),(1,"#55869C","--"),(2,"#B86636","-.")]:
            idx=int(round(radius/2*(len(r)-1)))
            axes[1,j].plot(time,values[:,idx],color=color,ls=style,lw=1.2,label=f"r={radius} cm")
        axes[1,j].set_ylabel(label);axes[1,j].grid(True)
    axes[1,0].plot(environment[:,0]/3600,environment[:,1]-273.15,color=GRAY,ls=":",lw=1,label="环境温度")
    axes[1,0].legend(loc="lower right",fontsize=7,frameon=False);axes[1,1].legend(loc="lower left",fontsize=7,frameon=False)
    for ax,panel in zip(axes.flat,["(a)","(b)","(c)","(d)"]):
        ax.set_xlim(0,3);ax.set_xticks(np.arange(0,3.01,.5));ax.set_xlabel("时间 / h")
        ax.text(.04,.97,panel,transform=ax.transAxes,va="top",bbox={"facecolor":"white","alpha":.8,"edgecolor":"none","pad":1})
    save(fig,folder,"fig4_q2_evolution")

def convergence_figure(records,folder):
    setup();fig,axes=plt.subplots(2,2,figsize=(6.3,4.8),layout="constrained")
    for row,key,unit in [(0,"T","°C"),(1,"C","kg/kg")]:
        for col,kind in enumerate(["space","time"]):
            ax=axes[row,col]
            for q,color in [(1,ORANGE),(2,BLUE)]:
                selected=[x for x in records if x["q"]==q and x["kind"]==kind]
                x=[v["fine"]["N"] if kind=="space" else v["fine"]["dt"] for v in selected]
                y=[v[key]["maximum"] for v in selected]
                if x:ax.semilogy(x,y,"o-" if q==1 else "s--",ms=3,lw=1.2,color=color,label=f"问题{q}")
            ax.axhline(1e-5,color=GRAY,lw=.9,ls=":",label="单项验收线")
            if kind=="space":
                ax.set_xticks([960,1280,1600],["960","1280","1600"])
            else:
                ax.set_xticks([1/1024,1/768],["1/1024","1/768"])
            ax.set_xlabel("径向单元数" if kind=="space" else "时间步尺度 τ / s")
            ax.set_ylabel(f"{('温度' if key=='T' else '含水率')}最大加密差异 / ({unit})")
            ax.grid(True,which="major");ax.legend(frameon=False,fontsize=7)
    save(fig,folder,"validation_space_time")
