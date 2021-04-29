const express = require("express");
const { spawn } = require("child_process");
const app = express();
const port = 6969;

app.get("/", (req, res) => {
  var dataToSend;

  const python = spawn("python", ["script1.py"]);

  python.stdout.on("data", function (data) {
    console.log("Pipe data from python script ...");
    dataToSend = data.toString();
    console.log({ dataToSend });
  });

  python.on("close", (code) => {
    console.log(`child process closed all stdio with code ${code}`);
    console.log({ dataToSend });
    res.send(dataToSend);
  });
});

app.listen(port, () =>
  console.log(`Tick Classifier listening on port ${port}!`)
);
