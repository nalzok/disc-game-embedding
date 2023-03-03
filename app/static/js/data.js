const hexdigest = document.getElementById("hexdigest").value
const dump = await (await fetch(`/api/v1/embed?hexdigest=${hexdigest}`)).json()
const X = dump["features"]
const F = dump["payoff"]
const E = dump["embedding"]
const eigen = dump["eigen"]
const data = Array.from(eigen).map((_, i) => ({
    feature: X[i],
    embedding: E[i],
    eigen: eigen[i],
}))

export { dump, data }
