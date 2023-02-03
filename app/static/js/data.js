// const numSamples = 100
// const numFeatures = 5

// Row-major
// const X = Array.from({length: numSamples}, () => Array.from({length: numFeatures}, () => Math.random()))
// const F = Array.from({length: numSamples}, () => Array.from({length: numSamples}, () => Math.random()))
// const E = Array.from({length: numSamples}, () => Array.from({length: numSamples}, () => Math.random()))
// const eigen = Array.from({length: numSamples}, () => Math.random()).sort()

const hexdigest = document.getElementById("hexdigest").value;
const dump = await (await fetch(`/api/v1/embed?hexdigest=${hexdigest}`)).json();
const X = dump["features"];
const F = dump["payoff"];
const E = dump["embedding"];
const eigen = dump["eigen"];

export const data = Array.from(eigen).map((_, i) => ({
    feature: X[i],
    embedding: E[i],
}));