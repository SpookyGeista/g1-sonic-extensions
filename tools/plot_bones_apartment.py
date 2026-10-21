#!/usr/bin/env python3
"""Read-only building plan over a recorded SONIC trajectory, with crawl opening section."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from plot_bones_crawl_route import load_motion, digest, draw_segments, place_numbers, PAPER, INK
from plot_actual_bones_route import load_trial

REPO=Path(__file__).resolve().parents[1]


def plot(layout_path,motion_path,trial_path,output):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['path.simplify']=False
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Polygon

    output=Path(output).resolve()
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Choose a NEW image and provenance output')
    layout_path=Path(layout_path).resolve()
    layout=json.loads(layout_path.read_text())
    sys.path.insert(0,str(layout_path.parent))
    from layout_geometry import collision_boxes_from_layout,validate_layout
    validate_layout(layout)
    ref=load_motion(motion_path);trial=load_trial(ref,trial_path)
    expected={b['prim']:b for b in collision_boxes_from_layout(layout)}
    recorded={b['prim']:b for b in trial['scene'].get('collision_boxes',[])}
    scene_match=(set(expected)==set(recorded) and all(
        np.allclose(expected[p][k],recorded[p][k],atol=1e-6,rtol=0)
        for p in expected for k in ('aabb_min','aabb_max')))
    fig=plt.figure(figsize=(16,10),facecolor=PAPER)
    gs=fig.add_gridspec(2,2,width_ratios=[3.2,1],height_ratios=[1,1],left=.055,right=.975,top=.82,bottom=.105,wspace=.13,hspace=.12)
    ax=fig.add_subplot(gs[:,0]);notes=fig.add_subplot(gs[0,1]);section=fig.add_subplot(gs[1,1])
    ax.set_facecolor('#e8e6dd')
    for room in layout.get('rooms',[]):
        lo,hi=np.array(room['bounds_xy'])
        color=layout['materials'][room['floor_material']]['color']
        ax.add_patch(Rectangle(lo,*(hi-lo),facecolor=color,alpha=.27,edgecolor='none',zorder=0))
        xy=(lo+hi)/2
        if room['name']=='Entry':xy=[.45,2.05]
        elif room['name']=='MainCorridor':xy=[4.15,.7]
        elif room['name']=='UpperPassage':xy=[8.6,6.75]
        ax.text(*xy,room['label'],ha='center',va='center',fontsize=10,color='#686964',zorder=1)
    def box(center,size,yaw,color,z=3,hatch=None):
        center=np.asarray(center)[:2];size=np.asarray(size)[:2]
        a=np.radians(yaw);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*size/2
        ax.add_patch(Polygon(corners@rotation.T+center,closed=True,facecolor=color,edgecolor='#454d4f',linewidth=.6,zorder=z,hatch=hatch))
    for wall in layout['walls']:
        box(wall['center'],wall['size'],wall.get('yaw_degrees',0),
            '#bf8d42' if wall.get('material')=='crawl_frame' else '#697171')
    gate=None
    for portal in layout['portals']:
        prefix=f"/World/Course/Portals/{portal['name']}/"
        low=portal['spring_height']<.8
        if low:gate=portal
        for name in ('Left','Right'):
            b=expected[prefix+name];box(b['center'],b['size'],b['yaw_degrees'],'#b67b29' if low else '#39464c',4)
        lintel=expected[prefix+'Lintel']
        if low:
            box(lintel['center'],lintel['size'],lintel['yaw_degrees'],'#e2b263',3,'////')
        else:
            a=np.radians(portal.get('yaw_degrees',0));axis=np.array([np.cos(a),np.sin(a)])
            endpoints=np.asarray(portal['center_xy'])+np.array([-1,1])[:,None]*axis*portal['clear_width']/2
            ax.plot(*endpoints.T,color='#60797d',ls=':',lw=1,zorder=2)
    xy=trial['root'][:,:2]
    ax.plot(*ref['root'][:,:2].T,'--',color='#969e9b',lw=1,alpha=.65,zorder=4)
    draw_segments(ax,xy,trial['stages'])
    ax.scatter(*xy[0],s=70,c='#27865b',edgecolors='white',zorder=10)
    ax.text(xy[0,0]-.15,xy[0,1]-.4,'START',fontsize=10,fontweight='bold',color='#27865b')
    ax.scatter(*xy[-1],s=60,c='#a14a3b',marker='s',edgecolors='white',zorder=10)
    ax.annotate('ENDE',xy=xy[-1],xytext=(11.5,5.15),arrowprops={'arrowstyle':'-','color':'#a14a3b'},fontsize=10,color='#a14a3b',fontweight='bold')
    portals={p['name']:p for p in layout['portals']}
    for name,label,offset,xytext in (
        ('CoverDoor01','Tür 1 / Deckung',(0,.85),(.3,1.5)),
        ('CoverDoor02','Tür 2 / Deckung',(.73,0),(5.3,3.65))):
        point=np.asarray(portals[name]['center_xy'])+offset
        ax.annotate(label,xy=point,xytext=xytext,arrowprops={'arrowstyle':'-','color':INK},fontsize=10,color=INK)
    ax.set(xlim=(-1.8,12.6),ylim=(-1.8,7.55),xlabel='Welt X [m]',ylabel='Welt Y [m]')
    ax.set_aspect('equal');ax.set_xticks(np.arange(-1,13,1));ax.set_yticks(np.arange(-1,8,1));ax.grid(alpha=.12,zorder=-1)
    place_numbers(ax,xy,trial['stages'])
    for spine in ax.spines.values():spine.set_color('#bfc5bd')
    notes.axis('off')
    notes.text(0,1,'Deine Raumidee',fontsize=16,color=INK,fontweight='bold',va='top')
    notes.text(0,.84,'• Zusammenhängende Räume\n  statt freier Kreuzwände\n\n• Erste Deckung an Tür 1,\n  danach durch dieselbe Öffnung\n\n• Zweite Deckung an Tür 2,\n  danach in den oberen Flur\n\n• Niedriges Tor im Krabbelteil\n  um etwa 35–36 Sekunden\n\n• Dach offen, keine Möbel\n  und keine Motion-Änderung',fontsize=11,color=INK,va='top',linespacing=1.35)
    if gate is None:raise ValueError('No low crawl portal')
    width,height,trim=gate['clear_width'],gate['spring_height'],gate['trim_width']
    section.set_facecolor(PAPER)
    for x in (-width/2-trim,width/2):
        section.add_patch(Rectangle((x,0),trim,height,fc='#b67b29',ec='#755022'))
    section.add_patch(Rectangle((-width/2-trim,height),width+2*trim,trim,fc='#d4a153',ec='#755022'))
    section.axhline(0,color='#555f60',lw=2)
    section.annotate('',xy=(-width/2-.25,height),xytext=(-width/2-.25,0),arrowprops={'arrowstyle':'<->','color':INK})
    section.text(-width/2-.33,height/2,f'{height:.2f} m frei',rotation=90,ha='right',va='center',fontsize=11,color=INK)
    section.annotate('',xy=(-width/2,-.1),xytext=(width/2,-.1),arrowprops={'arrowstyle':'<->','color':INK})
    section.text(0,-.19,f'{width:.2f} m lichte Breite',ha='center',fontsize=11,color=INK)
    section.set_title('Kriechtor · Frontansicht',loc='left',fontsize=14,fontweight='bold',color=INK)
    section.set(xlim=(-1.42,1.23),ylim=(-.45,.86));section.set_aspect('equal');section.axis('off')
    fig.suptitle('Gebäude um den G1-Laufweg',fontsize=24,fontweight='bold',color=INK,y=.965)
    caption='Gemessener SONIC-Lauf in dieser kollisionsaktiven Szene' if scene_match else 'Plan auf Basis eines bisherigen Isaac-Laufs — neue Szene noch nicht durch diesen Trace bestätigt'
    fig.text(.055,.908,caption,fontsize=12,color=INK)
    fig.text(.055,.872,'Farbig: gemessene Beckenbahn · Grau gestrichelt: Referenz · Dunkel: Wände/Türpfosten · Ocker: 50-cm-Kriechtor',fontsize=11,color='#697672')
    result=trial['summary']
    ground=result.get('ground_tracking',{}).get('ground_reference_tracking',{})
    status='Tracking-Prüfungen bestanden.' if result.get('passed') else 'Tracking-Prüfgrenze nicht vollständig bestanden; kein Kollisionsfreiheitsversprechen.'
    fig.text(.055,.045,status+' Die Beckenlinie allein ist kein Körperumriss.',fontsize=10,color='#9e5228')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180,facecolor=PAPER);plt.close(fig)
    hashes={**trial['source_sha256'],str(layout_path):digest(layout_path)}
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed while plotting')
    report={'layout':str(layout_path),'trial':str(trial_path),'source_sha256':hashes,
            'scene_geometry_matches_trace':bool(scene_match),'trial_passed':result.get('passed'),
            'crawl_gate':gate,'ground_tracking':ground,'no_spatial_alignment_or_smoothing':True,
            'image':str(output)}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:report[k] for k in ('image','scene_geometry_matches_trace','trial_passed')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',type=Path,required=True);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(plot(a.layout,a.motion,a.trial,a.output),indent=2))
