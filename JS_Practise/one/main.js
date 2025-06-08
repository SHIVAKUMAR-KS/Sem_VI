// setTimeout(() => console.log("Hi"), 1000);
// setInterval(() => console.log("Every 2s"), 2000);

const promise = new Promise((resolve, reject) => {
  resolve("Success");
});

promise.then(console.log).catch(console.error);
