module.exports = function (readFrames) {
  const assert = require("node:assert/strict");
  const { StringDecoder } = require("node:string_decoder");
  const expected = [
    { t: "reply", id: "w1", ok: ["j", "历史🙂e\u0301\n\"\\\u0000"] },
    { t: "settle", id: 2, ok: ["j", [false, 0, null]] },
    { t: "interrupt" },
  ];
  const bytes = Buffer.from("\n \t\r\n" + expected.map((frame) => JSON.stringify(frame)).join("\r\n") + "\n");
  const sizes = [1, 2, 3, 7, 64, bytes.length];
  for (const size of sizes) {
    const received = [];
    const read = readFrames((frame) => received.push(frame));
    const decoder = new StringDecoder("utf8");
    for (let at = 0; at < bytes.length; at += size) {
      read(decoder.write(bytes.subarray(at, at + size)));
    }
    read(decoder.end());
    assert.deepEqual(received, expected);
  }

  const received = [];
  const read = readFrames((frame) => received.push(frame));
  read('{"t":"reply","ok":"part');
  read("");
  read('ial"}');
  assert.deepEqual(received, []);
  read('\n{"t":"interrupt"}\n{"t":"shut');
  assert.deepEqual(received, [{ t: "reply", ok: "partial" }, { t: "interrupt" }]);
  read('down"}\n');
  assert.deepEqual(received.at(-1), { t: "shutdown" });

  const other = [];
  const independent = readFrames((frame) => other.push(frame));
  read('{"t":"inter');
  independent('{"t":"shutdown"}\n');
  read('rupt"}\n');
  assert.deepEqual(other, [{ t: "shutdown" }]);
  assert.deepEqual(received.at(-1), { t: "interrupt" });
  assert.throws(() => read("invalid JSON\n"), SyntaxError);

  const body = "历史🙂e\u0301\\\"\n".repeat(262144);
  const large = JSON.stringify({ t: "reply", ok: body }) + '\n{"t":"interrupt"}\n';
  const frames = [];
  const fragmented = readFrames((frame) => frames.push(frame));
  for (let at = 0; at < large.length; at += 16384) fragmented(large.slice(at, at + 16384));
  assert.deepEqual(frames, [{ t: "reply", ok: body }, { t: "interrupt" }]);
  return { partitions: sizes.length, frames: frames.length, characters: body.length };
};
