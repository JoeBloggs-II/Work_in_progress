// Monaco Editor
require.config({ paths: { 'vs': 'lib/monaco/min/vs' }});
require(['vs/editor/editor.main'], function() {
    monaco.editor.create(document.getElementById('editor'), {
        value: `function hello() {\n  console.log("Hello, world!");\n}`,
        language: 'javascript',
        theme: 'vs-dark'
    });
});

// xterm.js
const term = new Terminal();
term.open(document.getElementById('terminal'));
term.write('Welcome to your minimal web IDE!\r\n');
term.write('No backend yet, so commands won’t run.\r\n');
