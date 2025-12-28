let ctx=document.getElementById('board').getContext('2d');
let chart=new Chart(document.getElementById('chart'),{
  type:'line',
  data:{labels:[],datasets:[
    {label:'Hawks',borderColor:'red',data:[]},
    {label:'Doves',borderColor:'blue',data:[]}
  ]}
});

function start(){
  fetch('/api/game/start?hawks=20&doves=20',{method:'POST'});
}

function step(){
  fetch('/api/game/step').then(r=>r.json()).then(draw);
}

function draw(s){
  ctx.clearRect(0,0,600,600);
  ctx.beginPath();
  ctx.arc(300,300,280,0,2*Math.PI);
  ctx.stroke();
  s.creatures.forEach(c=>{
    ctx.beginPath();
    ctx.arc(300+c.x,300+c.y,5,0,2*Math.PI);
    ctx.fillStyle=c.type==='HAWK'?'red':'blue';
    ctx.fill();
  });
  chart.data.labels.push(s.day);
  chart.data.datasets[0].data.push(s.hawks);
  chart.data.datasets[1].data.push(s.doves);
  chart.update();
}
