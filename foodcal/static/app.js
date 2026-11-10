'use strict';
const $ = id => document.getElementById(id);
let info, file = null, previewURL = null, busy = false;
function message(text, bad=false) {$('message').textContent=text; $('message').className=bad?'error':'';}
async function updateInfo() {
  try {
    const response=await fetch('/api/info'); if(!response.ok)throw Error('无法读取服务状态');
    const data=await response.json();
    if(!info){for(const food of data.classes){const option=new Option(food.name+(food.nutrition_ready?'':'（营养值待补充）'),food.id);$('food').add(option);}}
    info=data; let status=data.model_message;
    if(data.data_preparation?.state==='downloading')status+=' 数据准备：'+data.data_preparation.percent+'%。';
    if(data.data_preparation?.state==='prepared')status+=' 训练数据已准备好。';
    $('status').textContent=status; $('recognize').disabled=busy||!file||!info.model_ready;
  }catch(error){message(error.message,true);}
}
async function post(url,payload){
  if(!info)throw Error('服务尚未就绪');
  const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Food-Token':info.token},body:JSON.stringify(payload)});
  const data=await response.json();if(!response.ok)throw Error(data.error||'处理失败');return data;
}
$('photo').addEventListener('change',event=>{
  file=event.target.files[0]||null;if(previewURL)URL.revokeObjectURL(previewURL);
  $('predictions').replaceChildren();$('result').textContent='';$('sources').replaceChildren();
  if(file){previewURL=URL.createObjectURL(file);$('preview').src=previewURL;$('preview').hidden=false;$('placeholder').hidden=true;}
  else{$('preview').hidden=true;$('placeholder').hidden=false;}
  $('recognize').disabled=busy||!file||!info?.model_ready;
});
$('recognize').addEventListener('click',async()=>{
  if(!file||busy||!info?.model_ready)return;
  busy=true;$('recognize').disabled=true;message('正在识别…');
  try{
    if(file.size>8_000_000)throw Error('请选择小于8MB的图片');
    const encoded=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(file);});
    const result=await post('/api/predict',{image:encoded});$('predictions').replaceChildren();
    for(const item of result.predictions){const li=document.createElement('li');li.textContent=item.name_zh+' '+(item.probability*100).toFixed(1)+'%';$('predictions').append(li);}
    $('food').value=result.predictions[0].class_name;$('predictionNote').textContent=result.note;message('请选择或确认菜品后估算热量。');
  }catch(error){message(error.message,true);}finally{busy=false;$('recognize').disabled=!file||!info?.model_ready;}
});
$('estimate').addEventListener('click',async()=>{
  $('result').textContent='';$('note').textContent='';$('sources').replaceChildren();
  message('正在计算…');
  try{
    const grams=Number($('grams').value);if(!$('grams').value.trim()||!Number.isFinite(grams)||grams<=0)throw Error('请输入有效克数');
    const result=await post('/api/estimate',{class_name:$('food').value,grams});
    const low=result.kcal_low,high=result.kcal_high;
    $('result').textContent=(low===high?low.toFixed(1):low.toFixed(1)+'–'+high.toFixed(1))+' 千卡';
    $('note').textContent=result.note||'基于公开营养参考值，配方不同会有差异。';
    const urls=result.source_urls||result.sources||[];
    for(const [index,source] of urls.entries()){const url=typeof source==='string'?source:source.url;if(!url||!/^https?:\/\//.test(url))continue;const a=document.createElement('a');a.href=url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=result.reference_foods?.[index]||'营养数据来源';$('sources').append(a,document.createElement('br'));}
    message('已按输入份量计算。');
  }catch(error){message(error.message,true);}
});
updateInfo();setInterval(updateInfo,10000);
