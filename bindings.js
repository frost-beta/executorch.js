try {
  module.exports = require('./build/Release/executorch.node');
} catch (error) {
  if (error.code == 'MODULE_NOT_FOUND')
    module.exports = require('./build/Debug/executorch.node');
  else
    throw error;
}
