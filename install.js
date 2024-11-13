#!/usr/bin/env node

const packageJson = require('./package.json');

// Local developement.
if (packageJson.version === '0.0.1-dev')
  process.exit(0);

const fs = require('node:fs');
const zlib = require('node:zlib');
const {pipeline} = require('node:stream/promises');

const urlPrefix = 'https://github.com/frost-beta/executorch.js/releases/download';

main().catch((error) => {
  console.error('Error downloading node-mlx:', error);
  process.exit(1);
});

async function main() {
  const dir = `${__dirname}/build/Release`;
  fs.mkdirSync(dir, {recursive: true});

  const os = {darwin: 'mac', win32: 'win'}[process.platform] ?? process.platform;
  const arch = process.arch;
  const version = packageJson.version;

  let backend;
  if (packageJson.name.includes('-'))
    backend = packageJson.name.substring(packageJson.name.lastIndexOf('-') + 1);
  else
    backend = os == 'mac' ? 'mps' : 'xnnpack';

  const prefix = `${urlPrefix}/v${version}/executorch-${backend}-${os}-${arch}-release`;
  await download(`${prefix}.node.gz`, `${dir}/executorch.node`);
}

async function download(url, filename) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Failed to download ${url}, status: ${response.status}`);

  const gunzip = zlib.createGunzip();
  await pipeline(response.body, gunzip, fs.createWriteStream(filename));
}
