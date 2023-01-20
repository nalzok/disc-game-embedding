const numSamples = 100
const numFeatures = 5

// Row-major
// const X = Array.from({length: numSamples}, () => Array.from({length: numFeatures}, () => Math.random()))
// const F = Array.from({length: numSamples}, () => Array.from({length: numSamples}, () => Math.random()))
// const E = Array.from({length: numSamples}, () => Array.from({length: numSamples}, () => Math.random()))
// const eigen = Array.from({length: numSamples}, () => Math.random()).sort()

const X = await (await fetch("../data/RPS/rps_5_X.json")).json();
const F = await (await fetch("../data/RPS/rps_5_F.json")).json();
const E = await (await fetch("../data/RPS/rps_5_E.json")).json();
const eigen = await (await fetch("../data/RPS/rps_5_eigen.json")).json();

export const data = Array.from(eigen).map((_, i) => ({
    feature: X[i],
    embedding: E[i],
}));