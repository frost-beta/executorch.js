import Mocha from 'mocha';
import fs from 'node:fs';
import {parseArgs} from 'node:util';
import {pipeline} from 'node:stream/promises';

// Add files.
const mocha = new Mocha();
for (const f of fs.readdirSync(__dirname)) {
  if (f.endsWith('.spec.ts') || f.endsWith('.spec.js'))
    mocha.addFile(`${__dirname}/${f}`);
}

// Handle -g and -i CLI args.
const {values} = parseArgs({
  options: {
    grep: {type: 'string', short: 'g'},
    invert: {type: 'string', short: 'i'},
  }
});
if (values.grep) mocha.grep(values.grep);
if (values.invert) mocha.invert();

// Run.
downloadModels().then(() => mocha.run(process.exit));

// Download fixtures.
async function downloadModels() {
  const urls = [
    'https://huggingface.co/frost-beta/mobilenet-v2-executorch-cpu/resolve/main/mv2.pte',
    'https://huggingface.co/frost-beta/mobilenet-v2-executorch-mps/resolve/main/mv2_mps_float16.pte',
    'https://huggingface.co/frost-beta/mobilenet-v2-executorch-xnnpack/resolve/main/mv2_xnnpack_fp32.pte',
  ];
  const fixtures = `${__dirname}/fixtures`;
  if (!fs.existsSync(fixtures))
    fs.mkdirSync(fixtures);
  await Promise.all(urls.map(async (url) => {
    const fileName = url.substr(url.lastIndexOf('/'));
    const filePath = `${fixtures}/${fileName}`;
    if (fs.existsSync(filePath))
      return;
    const response = await fetch(url);
    if (!response.body)
      throw new Error(`Failed to download ${url}, status: ${response.status}`);
    await pipeline(response.body, fs.createWriteStream(filePath));
  }));
}
