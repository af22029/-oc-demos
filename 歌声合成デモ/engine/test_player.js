// index.html のプレイヤー部分を、DOM と Audio を模擬して動かす検証スクリプト
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync('D:/歌声合成デモ_高校生向け/web/index.html', 'utf8');
const body = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].pop()[1];

// ---- 最小限の DOM 模擬 ----
const mkEl = (id) => {
  const el = {
    id, _text: '', innerHTML: '', style: {}, files: [], value: '0',
    classList: { _s: new Set(), add(c){this._s.add(c);}, remove(c){this._s.delete(c);},
                 toggle(c,v){ v ? this._s.add(c) : this._s.delete(c); }, contains(c){return this._s.has(c);} },
    set textContent(v){ this._text = v; }, get textContent(){ return this._text; },
    addEventListener(){}, appendChild(){}, click(){ this.onclick && this.onclick(); },
    getBoundingClientRect(){ return {left:0, width:1000, top:0, height:100}; },
    setPointerCapture(){}, getContext(){ return new Proxy({}, {get:()=>()=>{}}); },
    querySelector(sel){ return store(id + '|' + sel); },
    querySelectorAll(sel){ return [store(id + '|' + sel)]; },
    get parentElement(){ return store(id + '|parent'); },
  };
  return el;
};
const els = {};
const store = (k) => (els[k] = els[k] || mkEl(k));
const document = {
  getElementById: (id) => store(id),
  querySelectorAll: () => [],
  createElement: () => mkEl('new'),
  body: { style: {} },
};
// ---- Audio 模擬（currentTime が実際に進む） ----
const audios = [];
class Audio {
  constructor(src){ this.src = src || ''; this.currentTime = 0; this.paused = true;
                    this.readyState = 4; this.preload = ''; audios.push(this); }
  addEventListener(){}
  play(){ this.paused = false; return Promise.resolve(); }
  pause(){ this.paused = true; }
  load(){}
}
const localStorage = { getItem: () => null, setItem: () => {} };
const location = { protocol: 'file:' };
const requestAnimationFrame = () => {};
const window = { SONGS: {
  songs: [{key:'t', title:'テスト', note:'', notes:3, tracks:{
    rule:  {wav:'a_rule.wav',  png:'a.png', label:'ルール', sec: 20, peaks:[1,2,3]},
    model: {wav:'a_model.wav', png:'b.png', label:'学習',   sec: 20, peaks:[1,2,3]},
  }}],
  reals: [], credit: 'test'
}};

eval(body + `
globalThis.__api = { get cur(){return cur;}, get playing(){return playing;},
                     get players(){return players;}, get pos(){return pos;},
                     switchTo, seek, pause, play };
`);
const A = globalThis.__api;

// ---- 検証 ----
function assert(name, cond, extra=''){
  console.log((cond ? '  OK   ' : '  NG   ') + name + (extra ? '  ' + extra : ''));
  if (!cond) process.exitCode = 1;
}
const set = window.SONGS.songs[0];
console.log('① 初期状態');
assert('先頭トラックが選択されている', A.cur && A.cur.method === 'rule', 'cur=' + (A.cur && A.cur.method));
assert('まだ再生していない', A.playing === false);

console.log('② ルールベースを再生');
els['a-rule|button'].onclick();
assert('再生中', A.playing === true);
assert('rule の音源が鳴っている', A.players.rule.paused === false);

console.log('③ 5.0秒まで進めてから「学習モデル」を押す');
A.players.rule.currentTime = 5.0;
els['a-model|button'].onclick();
assert('学習モデルに切り替わった', A.cur.method === 'model');
assert('再生は続いている', A.playing === true && A.players.model.paused === false);
assert('rule は止まっている', A.players.rule.paused === true);
assert('位置が引き継がれた（5.0秒）', Math.abs(A.players.model.currentTime - 5.0) < 0.05,
       'model.currentTime=' + A.players.model.currentTime);

console.log('④ さらに 12秒へシークしてからルールへ戻す');
A.players.model.currentTime = 12.0;
els['a-rule|button'].onclick();
assert('ルールに戻った', A.cur.method === 'rule');
assert('位置が引き継がれた（12.0秒）', Math.abs(A.players.rule.currentTime - 12.0) < 0.05,
       'rule.currentTime=' + A.players.rule.currentTime);
assert('再生は続いている', A.playing === true && A.players.rule.paused === false);

console.log('⑤ 停止 → 別トラックへ切り替え（停止中でも位置を保つ）');
A.pause();
assert('停止した', A.playing === false);
A.players.rule.currentTime = 12.0;
A.switchTo(set, 'model', 'a');
assert('停止したまま', A.playing === false && A.players.model.paused === true);
assert('位置は 12.0秒 のまま', Math.abs(A.pos - 12.0) < 0.05, 'pos=' + A.pos);

console.log('⑥ シークバー操作');
A.seek(3.5);
assert('A.seek(3.5) が反映される', Math.abs(A.players.model.currentTime - 3.5) < 0.05,
       'currentTime=' + A.players.model.currentTime);
