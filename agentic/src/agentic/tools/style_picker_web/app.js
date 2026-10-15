'use strict';

const $ = (id) => document.getElementById(id);
const categoryNames = {'3D & Render':'3D 與渲染','Anime & Manga':'動漫與漫畫','Cartoon & Animation':'卡通與動畫','Character & Concept':'角色與概念','Cinema & Lighting':'電影與光影','Comics & Graphic Novels':'漫畫與圖像小說','Commercial & Editorial':'商業與編輯','Craft & Material':'手作與材質','Design & Posters':'設計與海報','Drawing & Sketch':'繪畫與速寫','Gaming':'遊戲','Illustration':'插畫','Painting & Fine Art':'繪畫與藝術','Photography':'攝影','Printmaking & Graphic Art':'版畫與平面藝術'};
let data, chosen = new Set(), saved = new Set(), visible = [], viewerId = null, saving = false;
const cards = new Map();
const draftKey = 'mediaoverload-krea2-style-draft';

function difference() {
  return new Set([...chosen, ...saved].filter((id) => chosen.has(id) !== saved.has(id))).size;
}
function message(text) { $('save-status').textContent = text; }
function showError(text) { $('error').textContent = text; $('error').hidden = !text; }
function rememberDraft() {
  try {
    if (difference()) localStorage.setItem(draftKey, JSON.stringify({revision:data.revision, ids:[...chosen]}));
    else localStorage.removeItem(draftKey);
  } catch (_) { /* The saved server config remains authoritative if browser storage is unavailable. */ }
}
function updateSummary() {
  const count = difference();
  $('selected-count').textContent = chosen.size;
  $('total-count').textContent = `/ ${data.styles.length} 已選`;
  $('excluded-count').textContent = `已取消 ${data.styles.length - chosen.size}`;
  $('save').disabled = saving || !count || !chosen.size;
  if (!saving) message(!chosen.size ? '至少保留一種風格才能儲存' : count ? `${count} 項變更尚未儲存` : '已套用至隨機生成');
}
function setChosen(id, checked) {
  if (checked) chosen.add(id); else chosen.delete(id);
  syncCard(id);
  if (viewerId === id) $('viewer-selected').checked = checked;
  rememberDraft();
  updateSummary();
  if ($('selection-filter').value !== 'all') filter();
}
function syncCard(id) {
  const card = cards.get(id);
  card.classList.toggle('is-selected', chosen.has(id));
  card.querySelector('input').checked = chosen.has(id);
}
function sceneInfo() { return data.scenes.find((scene) => scene.id === $('scene').value); }
function imageUrl(style) {
  const path = style.previews[$('scene').value];
  return path ? new URL(path, data.preview_base_url).href : null;
}
function openViewer(id) {
  viewerId = id;
  const style = data.styles.find((row) => row.id === id);
  $('viewer-name').textContent = style.name;
  $('viewer-category').textContent = style.categories.map((name) => categoryNames[name] || name).join(' / ');
  $('viewer-descriptor').textContent = style.descriptor;
  $('viewer-selected').checked = chosen.has(id);
  $('viewer-image').alt = `${style.name} · ${sceneInfo().label}`;
  const url = imageUrl(style);
  if (url) $('viewer-image').src = url;
  else $('viewer-image').removeAttribute('src');
  $('viewer').showModal();
}
function makeCard(style, index) {
  const card = document.createElement('article');
  card.className = 'style-card';
  card.dataset.styleId = style.id;
  const preview = document.createElement('button');
  preview.className = 'preview';
  preview.setAttribute('aria-label', `預覽 ${style.name}`);
  const img = document.createElement('img');
  img.loading = 'lazy'; img.decoding = 'async'; img.referrerPolicy = 'no-referrer';
  img.width = 1536; img.height = 1024;
  img.addEventListener('error', () => preview.classList.add('failed'));
  img.addEventListener('load', () => preview.classList.remove('failed'));
  const error = document.createElement('span');
  error.className = 'image-error'; error.textContent = '預覽未載入，點開仍可查看風格描述。';
  preview.append(img, error);
  preview.addEventListener('click', () => openViewer(style.id));
  const info = document.createElement('div'); info.className = 'card-info';
  const label = document.createElement('label');
  const input = document.createElement('input'); input.type = 'checkbox'; input.className = 'style-checkbox';
  input.setAttribute('aria-label', `保留 ${style.name}`);
  input.addEventListener('change', () => setChosen(style.id, input.checked));
  const title = document.createElement('span'); title.className = 'style-name'; title.textContent = style.name;
  const categories = document.createElement('span'); categories.className = 'style-category';
  categories.textContent = style.categories.map((name) => categoryNames[name] || name).join(' · ');
  title.append(categories); label.append(input, title);
  const ordinal = document.createElement('span'); ordinal.className = 'ordinal'; ordinal.textContent = String(index + 1).padStart(3, '0');
  info.append(label, ordinal); card.append(preview, info);
  cards.set(style.id, card);
  return card;
}
function filter() {
  const query = $('search').value.trim().toLocaleLowerCase();
  const category = $('category').value;
  const selection = $('selection-filter').value;
  visible = data.styles.filter((style) => {
    const text = [style.name, style.descriptor, ...style.categories, ...style.categories.map((c) => categoryNames[c] || c)].join(' ').toLocaleLowerCase();
    return (!query || text.includes(query)) && (!category || style.categories.includes(category))
      && (selection !== 'selected' || chosen.has(style.id)) && (selection !== 'excluded' || !chosen.has(style.id));
  });
  const ids = new Set(visible.map((style) => style.id));
  for (const [id, card] of cards) card.hidden = !ids.has(id);
  $('result-count').textContent = `顯示 ${visible.length} / ${data.styles.length} 種風格`;
  $('empty').hidden = !!visible.length;
  $('select-visible').disabled = $('clear-visible').disabled = !visible.length;
}
function changeScene() {
  const scene = sceneInfo();
  $('catalog').classList.toggle('is-portrait', scene.height > scene.width);
  $('catalog').classList.toggle('is-square', scene.height === scene.width);
  for (const style of data.styles) {
    const preview = cards.get(style.id).querySelector('.preview');
    const img = preview.querySelector('img');
    const url = imageUrl(style);
    img.alt = `${style.name} · ${scene.label}`;
    preview.classList.toggle('failed', !url);
    if (url) img.src = url; else img.removeAttribute('src');
  }
}
function bulk(rows, checked) {
  for (const style of rows) {
    if (checked) chosen.add(style.id); else chosen.delete(style.id);
    syncCard(style.id);
  }
  rememberDraft(); updateSummary(); filter();
}
async function save() {
  saving = true; updateSummary(); message('正在儲存…'); showError('');
  try {
    const response = await fetch('/api/selection', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({enabled_ids:[...chosen], revision:data.revision, token:data.token})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '儲存失敗，請重試。');
    data.revision = result.revision; saved = new Set(result.enabled_ids);
    rememberDraft();
  } catch (error) { showError(error.message); }
  finally { saving = false; updateSummary(); }
}
async function init() {
  try {
    const response = await fetch('/api/styles');
    data = await response.json();
    if (!response.ok) throw new Error(data.error || '無法讀取風格設定。');
    chosen = new Set(data.enabled_ids); saved = new Set(data.enabled_ids);
    try {
      const draft = JSON.parse(localStorage.getItem(draftKey));
      const ids = new Set(data.styles.map((style) => style.id));
      if (draft && draft.revision === data.revision && Array.isArray(draft.ids) && draft.ids.every((id) => ids.has(id))) chosen = new Set(draft.ids);
    } catch (_) { /* A malformed browser draft never replaces the saved config. */ }
    const categories = [...new Set(data.styles.flatMap((style) => style.categories))].sort();
    for (const name of categories) $('category').add(new Option(categoryNames[name] || name, name));
    for (const scene of data.scenes) $('scene').add(new Option(scene.label, scene.id));
    $('scene').value = 'animal-landscape';
    const fragment = document.createDocumentFragment();
    data.styles.forEach((style, index) => fragment.append(makeCard(style, index)));
    $('catalog').append(fragment);
    for (const style of data.styles) syncCard(style.id);
    $('select-all').textContent = `全選 ${data.styles.length} 種`;
    $('select-all').disabled = $('clear-all').disabled = false;
    $('loading').hidden = true;
    changeScene(); filter(); updateSummary();
    $('search').addEventListener('input', filter);
    $('category').addEventListener('change', filter);
    $('selection-filter').addEventListener('change', filter);
    $('scene').addEventListener('change', changeScene);
    $('select-all').addEventListener('click', () => bulk(data.styles, true));
    $('clear-all').addEventListener('click', () => bulk(data.styles, false));
    $('select-visible').addEventListener('click', () => bulk(visible, true));
    $('clear-visible').addEventListener('click', () => bulk(visible, false));
    $('save').addEventListener('click', save);
    $('viewer-selected').addEventListener('change', () => setChosen(viewerId, $('viewer-selected').checked));
    $('close-viewer').addEventListener('click', () => $('viewer').close());
    $('viewer').addEventListener('click', (event) => { if (event.target === $('viewer')) $('viewer').close(); });
    window.addEventListener('beforeunload', (event) => { if (difference()) { event.preventDefault(); event.returnValue = ''; } });
  } catch (error) { $('loading').hidden = true; showError(`圖庫無法載入：${error.message} 請重新整理頁面。`); }
}
init();
