import * as THREE from 'three';

const TAU=Math.PI*2;
const smooth=x=>{x=THREE.MathUtils.clamp(x,0,1);return x*x*x*(x*(x*6-15)+10);};
const qEuler=(x=0,y=0,z=0)=>new THREE.Quaternion().setFromEuler(new THREE.Euler(x,y,z,'XYZ'));
const v=(x=0,y=0,z=0)=>new THREE.Vector3(x,y,z);
function heldGlance(time){
  const keys=[[0,0],[2,0],[3.1,-.065],[5.5,-.065],[7,.042],
    [10,.042],[11.3,0],[16,0]],phase=time%16;
  for(let i=1;i<keys.length;i++)if(phase<=keys[i][0]){
    const a=keys[i-1],b=keys[i];
    return THREE.MathUtils.lerp(a[1],b[1],smooth((phase-a[0])/(b[0]-a[0])));
  }
  return 0;
}

/**
 * A local, skeleton-sized animation layer. It edits normalized humanoid pose
 * only. Original geometry, skin weights, material and spring settings survive.
 * World-space stance anchors are stationary during each support interval.
 */
export class NaturalMotion {
  constructor(vrm){
    this.vrm=vrm;this.h=vrm.humanoid;this.nodes={};this.rest={};
    this.h.resetNormalizedPose();vrm.update(0);vrm.scene.updateMatrixWorld(true);
    for(const [name,value]of Object.entries(this.h.normalizedHumanBones)){
      const node=value.node;this.nodes[name]=node;
      this.rest[name]={position:node.position.clone(),quaternion:node.quaternion.clone(),
        world:node.getWorldPosition(v()),worldQ:node.getWorldQuaternion(new THREE.Quaternion())};
    }
    this.legs={};
    for(const side of ['left','right']){
      const hip=this.rest[side+'UpperLeg'].world,knee=this.rest[side+'LowerLeg'].world,
        foot=this.rest[side+'Foot'].world;
      this.legs[side]={upper:hip.distanceTo(knee),lower:knee.distanceTo(foot),
        upperDir:knee.clone().sub(hip).normalize(),lowerDir:foot.clone().sub(knee).normalize(),
        ankleHeight:foot.y,restX:foot.x,restZ:foot.z};
    }
    this.mode='idle';this.modeStart=0;this.originZ=0;this.transition=null;
    this.period=1.52;this.stride=.42;this.stance=.62;this.lastContacts={};
    this.heightScale=(this.legs.left.upper+this.legs.left.lower)/.70019;
    this.stride*=this.heightScale;
    this.currentTravel=0;
  }
  capture(){
    return Object.fromEntries(Object.entries(this.nodes).map(([name,node])=>[name,{
      position:node.position.clone(),quaternion:node.quaternion.clone()
    }]));
  }
  setMode(mode,time,{transition=true}={}){
    if(!['idle','walk','greet'].includes(mode))throw Error('Unknown natural motion: '+mode);
    const previous=this.capture();
    const feet=Object.fromEntries(['left','right'].map(side=>{
      const foot=this.nodes[side+'Foot'];
      return[side,{position:foot.getWorldPosition(v()),
        pitch:new THREE.Euler().setFromQuaternion(foot.getWorldQuaternion(new THREE.Quaternion())).x}];
    }));
    this.originZ=this.nodes.hips.position.z-this.rest.hips.position.z;
    this.mode=mode;this.modeStart=time;this.transition=transition?{
      time,previous,feet,firstFoot:mode==='walk'?'left':
        feet.left.position.z<=feet.right.position.z?'left':'right'
    }:null;
  }
  local(name,x=0,y=0,z=0){
    if(this.nodes[name])this.nodes[name].quaternion.copy(this.rest[name].quaternion).multiply(qEuler(x,y,z));
  }
  worldRotation(name,rotation){
    const node=this.nodes[name];
    node.parent.updateWorldMatrix(true,false);
    const parentQ=node.parent.getWorldQuaternion(new THREE.Quaternion());
    node.quaternion.copy(parentQ.invert().multiply(rotation));
    node.updateWorldMatrix(false,true);
  }
  aim(name,restDirection,direction){
    this.worldRotation(name,new THREE.Quaternion().setFromUnitVectors(
      restDirection,direction.clone().normalize()).multiply(this.rest[name].worldQ));
  }
  solveLeg(side,target,footPitch=0){
    const data=this.legs[side],upper=side+'UpperLeg',lower=side+'LowerLeg';
    const hip=this.nodes[upper].getWorldPosition(v()),toward=target.clone().sub(hip);
    const desired=toward.length(),distance=THREE.MathUtils.clamp(desired,.02,data.upper+data.lower-.0001);
    toward.normalize();
    const pole=v(0,0,1).addScaledVector(toward,-toward.z).normalize();
    const along=(data.upper**2-data.lower**2+distance**2)/(2*distance);
    const knee=hip.clone().addScaledVector(toward,along)
      .addScaledVector(pole,Math.sqrt(Math.max(0,data.upper**2-along**2)));
    const actualTarget=hip.clone().addScaledVector(toward,distance);
    this.aim(upper,data.upperDir,knee.clone().sub(hip));
    this.aim(lower,data.lowerDir,actualTarget.clone().sub(knee));
    this.worldRotation(side+'Foot',qEuler(footPitch).multiply(this.rest[side+'Foot'].worldQ));
    this.local(side+'Toes',-Math.max(0,footPitch)*.45,0,0);
    return Math.max(0,desired-distance);
  }
  footAt(side,t){
    const data=this.legs[side],offset=side==='left'?0:.5;
    const cycle=t/this.period+offset,index=Math.floor(cycle),phase=cycle-index;
    const scale=this.heightScale,travel=this.stride*(index-offset);
    const anchorZ=this.originZ+data.restZ+travel+this.stride*this.stance/2;
    const x=data.restX*.88,pivotFront=.146*scale,pivotBack=-.061*scale;
    let target,pitch,contact=null;
    if(phase<this.stance){
      // Heel landing settles first; terminal stance pivots around the forefoot.
      pitch=-.09*(1-smooth(phase/.14))+.24*smooth((phase-.43)/(this.stance-.43));
      const pivot=v(0,-data.ankleHeight,pitch>=0?pivotFront:pivotBack);
      const worldPivot=v(x,0,anchorZ+pivot.z);
      target=worldPivot.clone().sub(pivot.clone().applyQuaternion(qEuler(pitch)));
      contact={side,cycle:index,phase,support:pitch<0?'heel':'forefoot',
        point:worldPivot.toArray(),localPivot:pivot.toArray()};
    }else{
      const u=(phase-this.stance)/(1-this.stance),ease=smooth(u);
      pitch=THREE.MathUtils.lerp(.24,-.09,ease)-.14*Math.sin(Math.PI*u);
      const endPivot=v(0,-data.ankleHeight,pivotBack);
      const end=v(x,0,anchorZ+this.stride+pivotBack)
        .sub(endPivot.clone().applyQuaternion(qEuler(-.09)));
      const startPivot=v(0,-data.ankleHeight,pivotFront);
      const start=v(x,0,anchorZ+pivotFront)
        .sub(startPivot.clone().applyQuaternion(qEuler(.24)));
      target=start.lerp(end,ease);
      target.y+=.036*scale*Math.sin(Math.PI*u)**1.3;
      target.x+=(side==='left'?1:-1)*.004*scale*Math.sin(Math.PI*u);
    }
    return{target,pitch,contact,phase};
  }
  apply(time){
    const t=Math.max(0,time-this.modeStart),scale=this.heightScale;
    for(const [name,node]of Object.entries(this.nodes)){
      node.position.copy(this.rest[name].position);node.quaternion.copy(this.rest[name].quaternion);
    }
    const hips=this.nodes.hips,cycle=TAU*t/this.period;
    const walking=this.mode==='walk',greeting=this.mode==='greet';
    const breath=walking?.3*Math.sin(cycle):Math.sin(TAU*t/4.7);
    const sway=walking?Math.sin(cycle):Math.sin(TAU*t/6.1);
    this.currentTravel=this.originZ+(walking?this.stride*t/this.period:0);
    hips.position.z+=this.currentTravel;
    hips.position.x+=scale*(walking?.010:.010)*sway;
    if(greeting)hips.position.x+=.003*scale;
    hips.position.y+=scale*(walking?-.019-.015*Math.cos(2*cycle):-.0045+.0007*breath);
    const pelvisYaw=walking?-.060*Math.cos(cycle):.011*Math.sin(t*.7);
    const pelvisRoll=walking?.025*sway:.025*sway;
    this.local('hips',walking?.018:.008,pelvisYaw,pelvisRoll);
    this.local('spine',-.010+.004*breath,-pelvisYaw*.44,-pelvisRoll*.40);
    this.local('chest',-.004+.007*breath,-pelvisYaw*.70,-pelvisRoll*.38);
    this.local('upperChest',.003*breath,walking?.007*Math.sin(cycle):.004*Math.sin(t*.6),-pelvisRoll*.16);
    this.local('neck',-.006,0,0);
    this.vrm.scene.updateMatrixWorld(true);
    this.worldRotation('head',qEuler(.004*Math.sin(walking?cycle:t*1.1),
      greeting?-.020+.025*Math.sin(t*.8):walking?.012*Math.sin(cycle):heldGlance(t),
      (greeting?.018:0)+.005*Math.sin(walking?cycle:t*.67))
      .multiply(this.rest.head.worldQ));
    this.lastContacts={};this.lastReachError=0;this.lastFootTargets={};
    for(const [side,sign]of [['left',1],['right',-1]]){
      const leg=this.legs[side];
      let target,pitch=0;
      if(walking){
        const foot=this.footAt(side,t);target=foot.target;pitch=foot.pitch;
        this.lastContacts[side]=foot.contact;
      }else target=v(leg.restX,leg.ankleHeight,leg.restZ+this.originZ);
      this.lastFootTargets[side]={position:target.clone(),pitch};
      this.lastReachError=Math.max(this.lastReachError,this.solveLeg(side,target,pitch));
      const armSwing=walking?-sign*Math.cos(cycle):.055*Math.sin(t*.8+sign*.8);
      const shoulderY=walking?sign*.016*Math.cos(cycle-.2):.004*breath;
      this.local(side+'Shoulder',0,shoulderY,-sign*(.012+.004*breath));
      this.vrm.scene.updateMatrixWorld(true);
      const wave=greeting&&side==='right';
      if(wave){
        const amplitude=.15*Math.sin(t*4.1);
        this.aim(side+'UpperArm',v(sign,0,0),v(sign*.78,-.30,.18));
        this.aim(side+'LowerArm',v(sign,0,0),v(sign*(.26+amplitude),.94,.12));
        // Distribute supination between forearm and wrist so the palm faces
        // the viewer instead of presenting a thin edge during the greeting.
        this.nodes[side+'LowerArm'].quaternion.multiply(qEuler(-.65,0,0));
        this.local(side+'Hand',-.92,sign*.035,sign*.10*Math.sin(t*4.1+.4));
      }else{
        this.aim(side+'UpperArm',v(sign,0,0),v(sign*.24,-.97,.27*armSwing));
        this.aim(side+'LowerArm',v(sign,0,0),v(sign*.12,-.97,.12+.34*armSwing));
        this.local(side+'Hand',.015,sign*.025,-sign*.025);
      }
      for(const [finger,index]of ['Index','Middle','Ring','Little'].map((f,i)=>[f,i])){
        const curl=wave?.025:.085+index*.016+.012*Math.sin(walking?cycle:t*.7);
        for(const [segment,multiplier]of [['Proximal',1],['Intermediate',1.25],['Distal',.72]]){
          this.local(side+finger+segment,0,0,-sign*curl*multiplier);
        }
      }
      this.local(side+'ThumbMetacarpal',0,-sign*.04,-sign*.025);
      this.local(side+'ThumbProximal',0,0,-sign*.05);
      this.local(side+'ThumbDistal',0,0,-sign*.04);
    }
    if(this.transition){
      const phase=THREE.MathUtils.clamp((time-this.transition.time)/.8,0,1);
      const weight=smooth(phase);
      for(const [name,node]of Object.entries(this.nodes)){
        const before=this.transition.previous[name];
        node.position.lerpVectors(before.position,node.position,weight);
        node.quaternion.slerpQuaternions(before.quaternion,node.quaternion,weight);
      }
      this.vrm.scene.updateMatrixWorld(true);
      // A whole-pose crossfade can push a shoe through the floor. Reposition
      // each foot in sequence and solve against the blended pelvis instead.
      const placements={};
      let pelvisLowering=0;
      for(const side of ['left','right']){
        const before=this.transition.feet[side],after=this.lastFootTargets[side];
        // Reposition one foot at a time. Overlapping both lift arcs creates
        // a small hop when stopping, even though neither sole penetrates.
        const footPhase=side===this.transition.firstFoot?Math.min(1,phase/.50):
          THREE.MathUtils.clamp((phase-.50)/.50,0,1);
        const blend=smooth(footPhase);
        const target=before.position.clone().lerp(after.position,blend);
        const distance=Math.hypot(after.position.x-before.position.x,after.position.z-before.position.z);
        target.y+=Math.min(.037*scale,distance*.40)*Math.sin(Math.PI*footPhase)**1.2;
        placements[side]={target,pitch:THREE.MathUtils.lerp(before.pitch,after.pitch,blend)};
        const hip=this.nodes[side+'UpperLeg'].getWorldPosition(v());
        const length=this.legs[side].upper+this.legs[side].lower-.0002;
        const horizontal=(hip.x-target.x)**2+(hip.z-target.z)**2;
        if(horizontal<length*length)
          pelvisLowering=Math.max(pelvisLowering,hip.y-target.y-Math.sqrt(length*length-horizontal));
      }
      // Blending the standing pelvis into a step may otherwise overextend
      // a leg by a few millimetres and pull its planted sole off the floor.
      this.nodes.hips.position.y-=Math.max(0,pelvisLowering);
      this.vrm.scene.updateMatrixWorld(true);
      this.lastReachError=0;
      for(const [side,{target,pitch}]of Object.entries(placements)){
        this.lastReachError=Math.max(this.lastReachError,
          this.solveLeg(side,target,pitch));
      }
      if(weight>=1)this.transition=null;
    }
    this.vrm.scene.updateMatrixWorld(true);
    return{travel:this.currentTravel,contacts:this.lastContacts,reachError:this.lastReachError};
  }
}
