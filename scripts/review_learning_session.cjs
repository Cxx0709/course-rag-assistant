// Run from the repository root. COURSE_RAG_PLAYWRIGHT can point to an existing
// Playwright package; no browser dependency is added to the application.
const {chromium} = require(process.env.COURSE_RAG_PLAYWRIGHT || 'playwright');
const {spawn} = require('node:child_process');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = process.cwd(), base = 'http://127.0.0.1:8031';
const phase = process.argv[2] || 'after';
assert(['before','after'].includes(phase));
let server, browser, logs = '';
(async () => {
  await fs.mkdir('.tmp/learning-agent-review', {recursive:true});
  const library = path.join(root, '.tmp', `agent-review-${phase}-${Date.now()}`);
  await fs.mkdir(library, {recursive:true});
  for (const entry of await fs.readdir('data/web', {withFileTypes:true})) {
    if(entry.isDirectory() && /^[a-f0-9]{32}$/.test(entry.name))
      await fs.cp(path.join('data/web',entry.name),path.join(library,entry.name),{recursive:true});
  }
  server = spawn(path.join(root,'.venv/Scripts/python.exe'), ['-c',
    'from pathlib import Path; import sys, uvicorn; from course_rag_assistant.web import create_app; uvicorn.run(create_app(Path(sys.argv[1]), generator_factory=lambda: None), host="127.0.0.1", port=8031)',library],
    {windowsHide:true,stdio:['ignore','pipe','pipe']});
  server.stdout.on('data', b=>logs+=b); server.stderr.on('data', b=>logs+=b);
  for(let i=0;i<100;i++) {
    if(server.exitCode!==null) throw Error(logs);
    try {if((await fetch(base+'/api/state')).ok) break;} catch {}
    await new Promise(r=>setTimeout(r,200));
  }
  browser=await chromium.launch({executablePath:process.env.COURSE_RAG_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1100}}), errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base);
  await page.getByRole('checkbox',{name:'使用资料：MLY-zh-cn.pdf',exact:true}).uncheck();
  await page.getByRole('checkbox',{name:'使用资料：d2l-zh-pytorch.pdf',exact:true}).check();
  const report={phase,reviewer:'assistant',blindAssessment:false,defaultLibraryModified:false,answers:[],browserErrors:errors};
  for(const question of ['什么是过拟合？','训练误差和泛化误差有什么区别？']) {
    await page.locator('#question').fill(question);
    const response=page.waitForResponse(r=>r.url().endsWith('/api/ask'));
    await page.locator('#ask-button').click();
    const answer=await (await response).json();
    if(phase==='after') {
      const text=answer.claims.map(c=>c.text).join('\n');
      assert(!answer.abstained);
      assert(answer.claims.every(c=>c.source_ids.length));
      assert(answer.sources.every(s=>s.name==='d2l-zh-pytorch.pdf' && s.page===158));
      if(question==='什么是过拟合？') assert(text.includes('现象称为过拟合'));
      else {
        assert(text.includes('训练数据集上计算得到的误差'));
        assert(text.includes('模型误差的期望'));
        assert(text.includes('原始样本的分布'));
        assert(!text.includes('2的可能性') && !text.startsWith('多。'));
      }
    }
    await page.locator('#source-list button').first().waitFor();
    await page.locator('#source-list button').first().click();
    await page.locator('#source-dialog').waitFor({state:'visible'});
    const url=await page.locator('#source-open').getAttribute('href');
    const source=await page.request.get(new URL(url,base).href,{headers:{Range:'bytes=0-4'}});
    assert([200,206].includes(source.status()));
    report.answers.push({question,answer,citationDialogVisible:true,sourceUrl:url,fileAvailable:true});
    await page.locator('#close-source').click();
  }
  const sectionsResponse=page.waitForResponse(r=>r.url().endsWith('/api/quiz/sections'));
  await page.locator('#quiz-mode').selectOption('section');
  const sections=(await (await sectionsResponse).json()).sections;
  const section=sections.find(s=>s.title==='4.4 模型选择、欠拟合和过拟合');
  assert(section); await page.locator('#quiz-section').selectOption(section.id);
  async function generate() {
    const response=page.waitForResponse(r=>r.url().endsWith('/api/quiz'));
    await page.locator('#quiz-create').click();
    const quiz=await (await response).json();
    await page.locator('#quiz-form').waitFor({state:'visible'});
    return quiz;
  }
  async function submit(quiz, terms) {
    for(const [i,q] of quiz.questions.entries()) {
      const index=q.options.indexOf(terms[i]); assert(index>=0);
      await page.locator(`input[name="quiz-${q.id}"][value="${index}"]`).check();
    }
    const response=page.waitForResponse(r=>r.url().endsWith('/grade'));
    await page.locator('#quiz-submit').click();
    const result=await (await response).json();
    await page.getByText(`本次得分 ${result.score} / ${result.total}`,{exact:true}).waitFor();
    return result;
  }
  async function review() {
    const response=page.waitForResponse(r=>r.url().endsWith('/api/review'));
    await page.locator('#review-load').click();
    return (await response).json();
  }
  const quiz=await generate(); assert.equal(quiz.questions.length,5);
  report.section=section; report.quiz=quiz;
  if(phase==='after') {
    assert(!quiz.questions.some(q=>/training error|generalization error/.test(q.prompt)));
    assert(quiz.questions[4].prompt.includes('对于固定的任务和数据分布'));
  }
  await page.locator('#quiz-form').screenshot({path:`.tmp/learning-agent-review/${phase}-desktop.png`});
  await page.setViewportSize({width:390,height:844});
  report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  assert(!report.mobileOverflow);
  await page.locator('#quiz-form').screenshot({path:`.tmp/learning-agent-review/${phase}-mobile.png`});
  await page.setViewportSize({width:1440,height:1100});
  report.chapterGrade=await submit(quiz,['训练误差','泛化误差','欠拟合','过拟合','数据集大小']);
  assert.equal(report.chapterGrade.score,5);
  report.reviewAfterCorrect=await review(); assert.equal(report.reviewAfterCorrect.gaps.length,0);
  await page.locator('#quiz-mode').selectOption('concept');
  await page.locator('#quiz-concept').fill('过拟合');
  const wrong=await generate();
  report.simulatedWrong=await submit(wrong,['欠拟合']);
  report.simulatedReview=await review();
  assert.equal(report.simulatedReview.gaps[0].concept,'过拟合');
  await page.locator('#review-list a').first().waitFor();
  const reviewUrl=await page.locator('#review-list a').first().getAttribute('href');
  assert(reviewUrl.endsWith('#page=167'));
  assert([200,206].includes((await page.request.get(new URL(reviewUrl,base).href,{headers:{Range:'bytes=0-4'}})).status()));
  await page.reload();
  await page.getByRole('checkbox',{name:'使用资料：MLY-zh-cn.pdf',exact:true}).uncheck();
  await page.locator('#quiz-concept').fill('过拟合');
  report.reviewAfterReload=await review(); assert.equal(report.reviewAfterReload.gaps.length,1);
  report.simulatedRetry=await submit(await generate(),['过拟合']);
  report.reviewAfterRetry=await review(); assert.equal(report.reviewAfterRetry.gaps.length,0);
  assert.deepEqual(errors,[]); report.passed=true;
  await fs.writeFile(`.tmp/learning-agent-review/${phase}.json`,JSON.stringify(report,null,2)+'\n');
  if(phase==='after') {
    await fs.copyFile('.tmp/learning-agent-review/after-desktop.png','docs/learning-session-review.png');
    await fs.writeFile('eval/reports/learning-session-browser-v1.json',JSON.stringify(report,null,2)+'\n');
  }
  console.log(JSON.stringify({phase,passed:true,chapterScore:report.chapterGrade.score,mobileOverflow:report.mobileOverflow,answers:report.answers.map(a=>({question:a.question,claims:a.answer.claims,sources:a.answer.sources.map(s=>({name:s.name,page:s.page}))})),browserErrors:errors},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(async()=>{await browser?.close();server?.kill();});
