import * as THREE from 'three';
import {OrbitControls} from '/public/vendor/OrbitControls.js';
import {PreviewAvatar} from '/public/preview-avatar.mjs';

const $ = id => document.getElementById(id);
const controlsUI = [...document.querySelectorAll('footer button, footer select')];
const viewport = $('viewport');
const avatarURL = new URL('../exports/avatar-preview.vrm', import.meta.url);
const walkURL = new URL('/public/elyth_base_walk_v1.vrma', location.origin);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#f5f4ef');
let renderer, avatar, observer;

function error(error) {
  $('overlay').hidden = false;
  $('status').textContent = '预览尚未完成';
  $('loading-title').textContent = '暂时无法显示角色';
  $('loading-detail').textContent = error?.message || '请通过项目的本地服务器打开此页面。';
  $('retry').hidden = false;
  controlsUI.forEach(control => control.disabled = true);
  renderer?.setAnimationLoop(null);
  console.error(error);
}
$('retry').addEventListener('click', () => location.reload());

async function getFile(url, name) {
  const response = await fetch(url, {cache:'no-store'});
  if (!response.ok) throw new Error(`${name} 读取失败（${response.status}）。`);
  return new File([await response.arrayBuffer()], name);
}

try {
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute('aria-label', '拖动可旋转的角色');
  viewport.prepend(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(34, 1, .01, 50);
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.dampingFactor = .07;
  scene.add(new THREE.HemisphereLight(0xfffbf4, 0xa7ac9e, 1.3));
  const key = new THREE.DirectionalLight(0xfff7e8, 2.1);
  key.position.set(-3, 5, 4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xe9efff, .65);
  fill.position.set(3, 2, 1);
  scene.add(fill);
  const floor = new THREE.Mesh(new THREE.CircleGeometry(1.5, 64),
    new THREE.MeshBasicMaterial({color:0xe9e9e1}));
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);
  avatar = new PreviewAvatar({scene});
  avatar.setActive(true);
  const [modelFile, walkFile] = await Promise.all([
    getFile(avatarURL, 'avatar-preview.vrm'), getFile(walkURL, 'standard-walk.vrma')
  ]);
  await avatar.loadVRM(modelFile);
  if (!avatar.vrm || !avatar.model) throw new Error('导出的文件缺少可用的 VRM Humanoid。');
  await avatar.loadVRMA(walkFile);
  if (!avatar.action) throw new Error('标准步行动作没有可播放的轨道。');
  avatar.model.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(avatar.model);
  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  if (bounds.isEmpty() || ![...size, ...center].every(Number.isFinite) || size.y <= 0)
    throw new Error('模型尺寸无效。');
  floor.position.set(center.x, bounds.min.y - .003, center.z);
  key.target.position.copy(center);
  scene.add(key.target);
  let frame = 'full', view = 'threequarter', resting = false;
  const viewButtons = [...document.querySelectorAll('[data-view]')];
  function preset(nextView = view) {
    view = nextView;
    const portrait = frame === 'portrait';
    const height = size.y * (portrait ? .40 : 1);
    const width = portrait ? size.y * .38 : size.x;
    const target = center.clone();
    if (portrait) target.y = bounds.max.y - size.y * .145;
    const tangent = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const distance = 1.16 * Math.max(height / (2 * tangent), width / (2 * tangent * camera.aspect));
    const direction = new THREE.Vector3(...({
      front:[0,.02,1], back:[0,.02,-1], threequarter:[.43,.06,1]
    }[view] || [0,0,1])).normalize();
    camera.position.copy(target).addScaledVector(direction, distance);
    orbit.target.copy(target);
    orbit.minDistance = size.y * .16;
    orbit.maxDistance = size.y * 7;
    orbit.update();
    viewButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
    $('portrait').setAttribute('aria-pressed', String(portrait));
  }
  function resize() {
    renderer.setSize(viewport.clientWidth, viewport.clientHeight, false);
    camera.aspect = viewport.clientWidth / Math.max(viewport.clientHeight, 1);
    camera.updateProjectionMatrix();
  }
  observer = new ResizeObserver(resize);
  observer.observe(viewport);
  resize();
  preset();
  function update() {
    $('walk').setAttribute('aria-pressed', String(!resting));
    $('pause').setAttribute('aria-pressed', String(!avatar.playing && !resting));
    $('pause').textContent = avatar.playing ? '暂停' : '继续';
    $('rest').setAttribute('aria-pressed', String(resting));
    $('physics').setAttribute('aria-pressed', String(avatar.physics));
    $('physics').textContent = avatar.physics ? '物理开启' : '物理关闭';
    $('status').textContent = resting ? 'T 姿势' : avatar.playing ? '步行中' : '动作已暂停';
  }
  $('walk').addEventListener('click', () => {
    resting = false; avatar.action.paused = false; avatar.play(); update();
  });
  $('pause').addEventListener('click', () => {
    if (resting) { resting = false; avatar.action.paused = false; avatar.play(); }
    else { avatar.playing = !avatar.playing; avatar.action.paused = !avatar.playing; }
    update();
  });
  $('rest').addEventListener('click', () => { avatar.stop(); resting = true; update(); });
  $('speed').addEventListener('change', () => { avatar.mixer.timeScale = Number($('speed').value); });
  $('physics').addEventListener('click', () => { avatar.setPhysics(!avatar.physics); update(); });
  $('reset').addEventListener('click', () => {
    avatar.model.updateMatrixWorld(true);
    avatar.simulationRemainder = 0;
    avatar.springManager?.reset();
  });
  viewButtons.forEach(button => button.addEventListener('click', () => preset(button.dataset.view)));
  $('portrait').addEventListener('click', () => { frame = frame === 'full' ? 'portrait' : 'full'; preset(); });
  orbit.addEventListener('start', () => viewButtons.forEach(button => button.setAttribute('aria-pressed', 'false')));
  renderer.domElement.addEventListener('webglcontextlost', event => {
    event.preventDefault(); error(new Error('3D 渲染连接已中断，请重新载入页面。'));
  });
  controlsUI.forEach(control => control.disabled = false);
  $('overlay').hidden = true;
  avatar.play();
  update();
  let last = performance.now(), statsAt = 0;
  renderer.setAnimationLoop(now => {
    const delta = Math.max(0, Math.min((now - last) / 1000, .1));
    last = now;
    avatar.tick(delta);
    orbit.update(delta);
    renderer.render(scene, camera);
    if (now - statsAt > 1000) {
      statsAt = now;
      $('stats').textContent = `${Math.round(renderer.info.render.triangles / 1000)}k 三角面 · ${renderer.info.render.calls} 次绘制 · ${avatar.springCount} 个物理关节`;
    }
  });
  document.addEventListener('visibilitychange', () => { last = performance.now(); });
  window.addEventListener('pagehide', () => {
    renderer.setAnimationLoop(null); observer.disconnect(); orbit.dispose();
    avatar.dispose(); floor.geometry.dispose(); floor.material.dispose(); renderer.dispose();
  }, {once:true});
} catch (failure) {
  error(failure);
}
