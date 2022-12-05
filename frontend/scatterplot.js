'use strict';
import {d3, parm, glob} from './util.js'

//console.log(glob.csvData);
//console.log(glob.csvData[test]);


const dataProc = () => {
  if (glob.csvData.plot1) {
    console.log('original test data', glob.csvData.plot1);
    glob.csvData.plot1data = glob.csvData.plot1.map(row => {
      const ret = {};
      ret.x = +row.x;
      ret.y = +row.y;
      return ret;
    });
  }
  if (glob.csvData.plot2) {
    console.log('original test data', glob.csvData.plot2);
    glob.csvData.plot2data = glob.csvData.plot2.map(row => {
      const ret = {};
      ret.x = +row.x;
      ret.y = +row.y;
      return ret;
    });
  }
  console.log(glob.csvData)
};

export const setup = () => {
    dataProc();
};
  
export const scatLabelPos = function () {
    // place the scatterplot axis labels.
    const marg = 30; // around the scatterplot domain
    const sz = parm.scatSize;
    /* since these two had style "position: absolute", we have to specify where they will be, and
    this is done relative to the previously placed element, the canvas */
    ['#scat-axes', '#scat-marks-container'].map((pid) =>
      d3
        .select(pid)
        .style('left', -marg)
        .style('top', -marg)
        .attr('width', 2 * marg + sz)
        .attr('height', 2 * marg + sz)
    );
    d3.select('#y-axis').attr('transform', `translate(${marg},${marg + sz / 2}) rotate(-90)`);
    d3.select('#x-axis').attr('transform', `translate(${marg + sz / 2},${marg + sz})`);
    d3.select('#scat-marks')
      .attr('transform', `translate(${marg},${marg})`)
      .attr('width', sz)
      .attr('height', sz);
};

/* scatMarksInit() creates the per-state circles to be drawn over the scatterplot */
export const scatMarksInit = function (id, data) {
    /* maps interval [0,data.length-1] to [0,parm.scatSize-1]; this is NOT an especially informative thing
    to do; it just gives all the tickmarks some well-defined initial location */
    const tscl = d3
      .scaleLinear()
      .domain([0, data.length - 1])
      .range([0, parm.scatSize]);
    /* create the circles */
    d3.select('#' + id)
      .selectAll('circle')
      .data(data)
      .join('circle')
      .attr('class', 'stateScat')
      // note that every scatterplot mark gets its own id, eg. 'stateScat_IL'
      .attr('id', d => `stateScat_${d.StateAbbr}`)
      .attr('r', parm.circRad)
      .attr('cx', (d, ii) => tscl(ii))
      .attr('cy', (d, ii) => parm.scatSize - tscl(ii));
  };




export const xScaleGen = function (data, width) {
    let ret = null;
    ret = d3.scaleLinear() 
    .domain([d3.min(data, function(d) {return d.x;}), d3.max(data, function(d) {return d.x;})])
    .range([0, width])
    //.padding(padby)
    console.log(data)
    return ret;
}
export var xScale = function (data, width, n) {
  let ret = null;
  ret = d3.scaleLinear() 
  .domain([d3.min(data, function(d) {return d.x;}), d3.max(data, function(d) {return d.x;})])
  .range([0, width])
  //.padding(padby)
  console.log(data)
  return ret;
}
export const yScaleGen = function (dataRange, height) {
    let ret = null;
    ret = d3.scaleLinear()
    .domain(dataRange)
    .range([height, 0])
    return ret;
}
export var yScale = function (dataRange, height, n) {
  let ret = null;
  ret = d3.scaleLinear()
  .domain(dataRange)
  .range([height, 0])
  return ret;
}
export const dots = function (svg, data, layout, radius) {
    svg.selectAll()
      .data(data)
      .join("circle")
      .attr("class", "dot") 
      .attr("cx", d => layout.xScale(d.x)) //+ 0.5 * layout.xScale.bandwidth()) 
      .attr("cy", d => layout.yScale(d.y))
      .attr("r", radius);
}