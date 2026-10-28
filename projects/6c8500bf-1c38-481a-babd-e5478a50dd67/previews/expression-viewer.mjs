import * as THREE from 'three';
import {OrbitControls} from '/public/vendor/OrbitControls.js';
import {PreviewAvatar} from '/public/preview-avatar.mjs';

const $ = id => document.getElementById(id);
const viewport = $('viewport');
const inputs = [...document.querySelectorAll('aside button, aside input, [data-angle]')];
const expressionButtons = [...document.querySelectorAll('[data-expression]')];
const angleButtons = [...document.querySelectorAll('[data-angle]')];
const labels = Object.fromEntries(expressionButtons.map(button =>
  [button.dataset.expression, button.textContent]));
const eyelids = new Set(['blink', 'blinkLeft', 'blinkRight']);
const vowels = ['aa', 'ih', 'ou', 'ee', 'oh'];
const moods = ['neutral', 'happy', 'angry', 'sad', 'relaxed', 'surprised'];
const projectURL = new URL('../', import.meta.url);
const modelURL = new URL('../exports/expression-review/avatar-preview.vrm', import.meta.url);
let renderer, avatar, orbit, observer;
let ready = false, selected = 'neutral', strength = 1, demoStart = null;
let autoBlink = !matchMedia('(prefers-reduced-motion: reduce)').matches, autoOrbit = false;
let time = 0, nextBlink = 2.4, blinkStarted = null, shownDemoLabel = '';
const scene = new THREE.Scene();
scene.background = new THREE.Color('#f5f4ef');

function toggleUI(id, value) {
  $(id).setAttribute('aria-pressed', String(value));
  $(id).textContent = value ? '开启' : '关闭';
}
toggleUI('autoblink', autoBlink);
function fail(failure) {
  ready = false;
  renderer?.setAnimationLoop(null);
  inputs.forEach(input => input.disabled = true);
  $('overlay').hidden = false;
  $('status').textContent = '预览待就绪';
  $('loading-title').textContent = '暂时无法显示表情模型';
  $('loading-detail').textContent = failure?.message || '请通过项目本地服务器打开此页面。';
  $('retry').hidden = false;
  console.error(failure);
}
$('retry').addEventListener('click', () => location.reload());
function stopDemo() {
  demoStart = null;
  $('demo').setAttribute('aria-pressed', 'false');
  $('demo').textContent = '播放表情与说话演示';
}
function choose(name) {
  stopDemo();
  selected = name;
  expressionButtons.forEach(button => button.setAttribute('aria-pressed',
    String(button.dataset.expression === name)));
  $('expression-name').textContent = labels[name];
  $('expression-detail').textContent = name === 'neutral' ? '默认面容' : `强度 ${Math.round(strength * 100)}%`;
  $('status').textContent = '表情已就绪';
}
expressionButtons.forEach(button => button.addEventListener('click', () => choose(button.dataset.expression)));
$('intensity').addEventListener('input', () => {
  strength = Number($('intensity').value) / 100;
  $('intensity-value').value = `${Math.round(strength * 100)}%`;
  if (demoStart === null) $('expression-detail').textContent =
    selected === 'neutral' ? '默认面容' : `强度 ${Math.round(strength * 100)}%`;
});
$('autoblink').addEventListener('click', () => {
  autoBlink = !autoBlink;
  blinkStarted = null;
  nextBlink = time + 2;
  toggleUI('autoblink', autoBlink);
});
$('orbit').addEventListener('click', () => {
  autoOrbit = !autoOrbit;
  toggleUI('orbit', autoOrbit);
});
$('demo').addEventListener('click', () => {
  if (demoStart !== null) { choose('neutral'); return; }
  demoStart = time;
  shownDemoLabel = '';
  $('demo').setAttribute('aria-pressed', 'true');
  $('demo').textContent = '结束演示';
  expressionButtons.forEach(button => button.setAttribute('aria-pressed', 'false'));
  $('status').textContent = '表情演示中';
});

function animationValues() {
  const values = {};
  if (demoStart !== null) {
    const t = time - demoStart;
    if (t >= 22) { choose('neutral'); return values; }
    let label;
    if (t < 15) {
      const index = Math.floor(t / 2.5);
      const phase = (t % 2.5) / 2.5;
      const envelope = THREE.MathUtils.smoothstep(phase, 0, .2)
        * (1 - THREE.MathUtils.smoothstep(phase, .8, 1));
      const mood = moods[index];
      if (mood !== 'neutral') values[mood] = strength * envelope;
      label = labels[mood];
    } else {
      const syllable = (t - 15) * 3.3;
      const index = Math.floor(syllable) % vowels.length;
      const phase = syllable % 1;
      const amount = Math.pow(Math.sin(phase * Math.PI), 1.4) * .85 * strength;
      values[vowels[index]] = amount;
      label = '连续口型';
    }
    if (label !== shownDemoLabel) {
      $('expression-name').textContent = label;
      $('expression-detail').textContent = '表情与说话演示';
      shownDemoLabel = label;
    }
  } else if (selected !== 'neutral') {
    values[selected] = strength;
  }
  if (autoBlink && !(demoStart === null && eyelids.has(selected))) {
    if (blinkStarted === null && time >= nextBlink) blinkStarted = time;
    if (blinkStarted !== null) {
      const elapsed = time - blinkStarted;
      const amount = elapsed < .085 ? elapsed / .085
        : elapsed < .13 ? 1 : Math.max(0, 1 - (elapsed - .13) / .135);
      values.blink = Math.max(values.blink || 0, Math.min(1, amount));
      if (elapsed >= .265) {
        blinkStarted = null;
        nextBlink = time + 2.4 + Math.random() * 2.7;
      }
    }
  }
  return values;
}

try {
  if (location.protocol === 'file:') throw Error('请通过项目本地服务器打开表情查看页面。');
  if (modelURL.origin !== location.origin || !modelURL.pathname.startsWith(projectURL.pathname)) {
    throw Error('表情模型必须来自当前本地项目。');
  }
  renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute('aria-label', '角色脸部，拖动旋转或使用观察角度按钮');
  viewport.prepend(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(30, 1, .01, 20);
  orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.dampingFactor = .09;
  orbit.minPolarAngle = Math.PI * .31;
  orbit.maxPolarAngle = Math.PI * .68;
  orbit.minAzimuthAngle = -Math.PI / 2;
  orbit.maxAzimuthAngle = Math.PI / 2;
  scene.add(new THREE.HemisphereLight(0xfffcf6, 0xc6c8bc, 1.2));
  const key = new THREE.DirectionalLight(0xfff8ee, 1.7);
  key.position.set(-2, 4, 4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xf0f4ff, .8);
  fill.position.set(3, 2, 2);
  scene.add(fill);
  avatar = new PreviewAvatar({scene});
  avatar.setActive(true);
  const response = await fetch(modelURL, {cache: 'no-store'});
  if (!response.ok) {
    if (response.status === 404) throw Error('新版表情预览尚未导出。完成导出后，点击“重新载入”即可查看。');
    throw Error(`表情模型读取失败（${response.status}）。`);
  }
  await avatar.loadVRM(new File([await response.arrayBuffer()], 'avatar-preview.vrm'));
  const manager = avatar.vrm?.expressionManager;
  if (!manager || !avatar.model) throw Error('模型中没有可用的表情设置。');
  const missing = Object.keys(labels).filter(name => !manager.getExpression(name));
  if (missing.length) throw Error('表情预览仍不完整：' + missing.map(name => labels[name]).join('、'));
  const quaternion = z => new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, z)).toArray();
  avatar.vrm.humanoid.setNormalizedPose({
    leftUpperArm: {rotation: quaternion(-1.06)}, rightUpperArm: {rotation: quaternion(1.06)},
    leftLowerArm: {rotation: quaternion(-.10)}, rightLowerArm: {rotation: quaternion(.10)},
  });
  manager.resetValues();
  if (manager.getExpression('blush')) manager.setValue('blush', 0);
  avatar.vrm.update(0);
  avatar.model.updateMatrixWorld(true);
  avatar.springManager?.reset();
  const bounds = new THREE.Box3().setFromObject(avatar.model);
  const height = bounds.max.y - bounds.min.y;
  if (!Number.isFinite(height) || height <= 0) throw Error('角色尺寸无效。');
  const target = bounds.getCenter(new THREE.Vector3());
  target.y = bounds.max.y - height * .105;
  const head = avatar.vrm.humanoid.getRawBoneNode('head');
  if (head) target.z = head.getWorldPosition(new THREE.Vector3()).z;
  let currentAngle = 0;
  let framingDistance = 1;
  function frame(angle = currentAngle) {
    currentAngle = angle;
    const tangent = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    framingDistance = 1.08 * Math.max(height * .29 / (2 * tangent),
      height * .30 / (2 * tangent * camera.aspect));
    const radians = THREE.MathUtils.degToRad(angle);
    camera.position.set(target.x + Math.sin(radians) * framingDistance,
      target.y + .012, target.z + Math.cos(radians) * framingDistance);
    orbit.target.copy(target);
    orbit.minDistance = height * .17;
    orbit.maxDistance = height * 1.8;
    orbit.update();
    angleButtons.forEach(button => button.setAttribute('aria-pressed',
      String(Number(button.dataset.angle) === angle)));
  }
  function resize() {
    const width = viewport.clientWidth, height = Math.max(viewport.clientHeight, 1);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  observer = new ResizeObserver(() => { resize(); if (ready) frame(currentAngle); });
  observer.observe(viewport);
  resize();
  frame();
  angleButtons.forEach(button => button.addEventListener('click', () => {
    autoOrbit = false;
    toggleUI('orbit', false);
    frame(Number(button.dataset.angle));
  }));
  orbit.addEventListener('start', () => {
    autoOrbit = false;
    toggleUI('orbit', false);
    angleButtons.forEach(button => button.setAttribute('aria-pressed', 'false'));
  });
  renderer.domElement.addEventListener('webglcontextlost', event => {
    event.preventDefault();
    fail(Error('3D 渲染连接已中断，请重新载入。'));
  });
  inputs.forEach(input => input.disabled = false);
  $('overlay').hidden = true;
  ready = true;
  choose('neutral');
  let previous = performance.now();
  renderer.setAnimationLoop(now => {
    const delta = Math.min(Math.max((now - previous) / 1000, 0), .05);
    previous = now;
    time += delta;
    manager.resetValues();
    for (const [name, value] of Object.entries(animationValues())) manager.setValue(name, value);
    if (manager.getExpression('blush')) manager.setValue('blush', 0);
    avatar.tick(delta);
    if (autoOrbit) {
      const angle = Math.sin(time * .22) * Math.PI / 4;
      camera.position.set(target.x + Math.sin(angle) * framingDistance,
        target.y + .012, target.z + Math.cos(angle) * framingDistance);
      orbit.target.copy(target);
    }
    orbit.update();
    renderer.render(scene, camera);
  });
  document.addEventListener('visibilitychange', () => { previous = performance.now(); });
  window.addEventListener('pagehide', () => {
    renderer.setAnimationLoop(null);
    observer.disconnect();
    orbit.dispose();
    avatar.dispose();
    renderer.dispose();
  }, {once: true});
} catch (failure) {
  fail(failure);
}
