let model;
const sliders = [];

const range1 = document.querySelector("#slider1")
const range2 = document.querySelector("#slider2")
const range3 = document.querySelector("#slider3")
const range4 = document.querySelector("#slider4")
const range5 = document.querySelector("#slider5")
const range6 = document.querySelector("#slider6")

const ranges = [range1, range2, range3, range4, range5, range6]

for (let i = 0; i < 6; i++) {

    ranges[i].addEventListener("input", () => {
	const min = ranges[i].min
	const max = ranges[i].max
	const currentVal = ranges[i].value

	ranges[i].style.backgroundSize = ((currentVal - min) / (max - min)) * 100 + "% 100%"
    } )
}





function loadData(name) {
    const request = new XMLHttpRequest();
    request.open('GET', name, false); // false makes the request synchronous
    request.send(null);

    if (request.status === 200) {
        return JSON.parse(request.responseText);
    } else {
        console.error('Failed to load data.json');
        return null;
    }
}

const tt_data = loadData('tt_data.json');
const te_data = loadData('te_data.json');
const ee_data = loadData('ee_data.json');
const pp_data = loadData('pp_data.json');


var data = null;
var checkboxChecked = false;



function resetSlidersToBestfit() {
    const bestfit = [0.0223, 0.119, 67.7, 3.04, 0.965, 0.054]

    for (let i = 0; i < 6; i++) {
	ranges[i].value = bestfit[i]
	ranges[i].style.backgroundSize = ((bestfit[i] - ranges[i].min) / (ranges[i].max - ranges[i].min)) * 100 + "% 100%"
    }
}

function resetGraphToBestfit() {
    resetSlidersToBestfit();
    updateDataAndChart();
}


const selectedOption = { value: "TT" };

// Function to handle dropdown selection
function handleSelection(event) {
    // Set the selected value in the object
    selectedOption.value = event.target.value;
    if (checkboxChecked) {
	if (selectedOption.value === "TT") {
            data = tt_data;
        } else if (selectedOption.value === "TE") {
            data = te_data;
        } else if (selectedOption.value === "EE") {
            data = ee_data;
        } else if (selectedOption.value === "PP") {
            data = pp_data;
        }
    };
    updateAxes();
    updateDataAndChart();
}

function showData(checkboxElem) {
    if (checkboxElem.checked) {
	if (selectedOption.value === "TT") {
	    data = tt_data;
	} else if (selectedOption.value === "TE") {
	    data = te_data;
	} else if (selectedOption.value === "EE") {
	    data = ee_data;
	} else if (selectedOption.value === "PP") {
	    data = pp_data;
	}
	checkboxChecked = true;
    } else {
	data = null;
	checkboxChecked = false;
    };
    updateDataAndChart();
}



window.addEventListener('load', async () => {
    resetSlidersToBestfit();
    const modelUrl = 'web_model/model.json'; // Adjust the path if necessary
    model = await tf.loadGraphModel(modelUrl);

    // Initialize sliders
    for (let i = 1; i <= 6; i++) {
        const slider = document.getElementById(`slider${i}`);
        sliders.push(slider);
        slider.addEventListener('input', updateDataAndChart);
        document.getElementById(`slider${i}-value`).innerText = slider.value;
    }
    document.body.addEventListener("click", updateAxesAndChart);

    window.addEventListener("resize", updateAxesAndChart);

    // Initial chart display
    updateAxes();
    updateDataAndChart();
});




// initialize dictionaries. One contains axis plot configurations, and the other contains the data to be plotted
const axisConfig = {
    chart: {
	type: 'spline',
	backgroundColor: '#FFFFFF',
	marginLeft: 80,
	marginRight: 30,
	marginTop: 15,
	plotBorderWidth: 2,
	plotBorderColor: 'black',
    },
    title: {
		text: ''
    },
    xAxis: {
	type: 'linear',
	tickPositions: [],
	labels: {
	    formatter: function() {
		return xTotalLabels[xTotalTicks.indexOf(this.value)] || '';
	    },
	    style: {
		color: 'black',
		fontSize: '20px'
	    },
	    rotation: 0,
	    overflow: 'allow',
	},
	title: {
	    text: `<img src="./svg_images/ell.svg" style="width:120px; height:50px;"/>`,
	    useHTML: true,
	    align: 'middle',
	},
	lineColor: 'black',
	lineWidth: 2,
	tickWidth: 2,
	tickLength: 10,
	tickColor: 'black',
	min: 0,
	max: 1,
	},
    yAxis: {
	tickPositions: [0, 2000, 4000, 6000, 8000],
	labels: {
	    formatter: function() {
		return ['0', '2000', '4000', '6000', '8000'].includes(this.value.toString()) ? this.value : '';
	    },
	    style: {
		color: 'black',
		fontSize: '20px'
	    },
	    rotation: 270,
	},
	title: {
	    text: `<img src="./svg_images/Cl.svg" style="position:absolute; width:150px; height:150px;"/>`,
	    useHTML: true,
	    rotation: 0,
	    align: 'middle',
	    margin: 100,
	    y: -70,
	},
	gridLineWidth: 0,
	lineWidth: 2,
	lineColor: 'black',
	tickWidth: 2,
	tickLength: 10,
	tickColor: 'black',
	alignTicks: false,
	endOnTick: false,
	startOnTick: false,
	min: 0,
	max: 8400,
    },
    series: [{
	data: [],
	lineWidth: 6,
	marker: {
	    enabled: false
	},
	color: 'rgb(38, 135, 242)'
    },{
	type: 'scatter',
	data: [],
	opacity: 0.6,
	marker: {
	    symbol: 'circle',
	    radius: 4,
	    fillColor: 'limeGreen',
	}
    },{
	type: 'errorbar',
	data: [],
	linkedTo: ':previous',
	stemWidth: 3,
	stemColor: 'limeGreen',
	whiskerLength: 0,
	opacity: 0.6,
	pointRange: 0,
	pointWidth: 3,
    }],
    legend: {
	enabled: false
    },
    tooltip: {
	enabled: false
    },
    credits: {
	enabled: false
    }
};


const logscale = 200.0;
const transition = 0.3;
const l_min = 2.0;

function generateXTicks(fewTicks = false) {
    let majorTicks = [];
    let minorTicks = [];
    let logOrder = 1;
    var linMajor = 500;
    var linMinor = 100;
    var minorTicksLogFactor = 1;
    if (fewTicks) {
	linMajor = 1000;
	linMinor = 200;
	minorTicksLogFactor = 2;
	}
    
    for (let i = 2; i <= 2500; i++) {
        if (i < logscale) {
            if (Number.isInteger(Math.log10(i))) {
                majorTicks.push(logScale(i));
                logOrder *= 10;
            } else if (i/minorTicksLogFactor % logOrder === 0) {
                minorTicks.push(logScale(i));
            }
        } else {
            if (i % linMajor === 0) {
                majorTicks.push(linScale(i));
            } else if (i % linMinor === 0) {
                minorTicks.push(linScale(i));
            }
        }
    }
    
    return { majorTicks, minorTicks };
}

function logScale(x) {
    const logx = Math.log10(x);
    const logxnew = (logx / Math.log10(logscale)) * transition;
    
    const loglmin = (Math.log10(l_min) / Math.log10(logscale)) * transition;
    const logscaleVal = (Math.log10(logscale) / Math.log10(logscale)) * transition;
    const logdiff = logscaleVal - loglmin;
    
    return ((logxnew - loglmin) * transition / logdiff);
}

function linScale(x) {
    const linxnew = (x / (2500 - logscale)) - (logscale / (2500 - logscale));
    return ((linxnew * (1 - transition)) + transition);
}

function scaleGraphPoint(xValue) {
    if (xValue <= logscale) {
        return logScale(xValue);
    } else {
        return linScale(xValue);
    }
}

// Function to merge, sort arrays based on values, and synchronize labels
function mergeSortAndSyncArrays(arr1, labels1, arr2, labels2) {
    // Combine arrays and labels into array of objects
    let combinedData = [];
    
    // Combine data from arr1 with labels from labels1
    for (let i = 0; i < arr1.length; i++) {
        combinedData.push({ value: arr1[i], label: labels1[i] });
    }
    
    // Combine data from arr2 with labels from labels2
    for (let i = 0; i < arr2.length; i++) {
        combinedData.push({ value: arr2[i], label: labels2[i] });
    }

    // Sort combined data based on value
    combinedData.sort((a, b) => a.value - b.value);

    // Extract sorted arrays and labels from sorted combined data
    let sortedValues = [];
    let sortedLabels = [];

    for (let item of combinedData) {
        sortedValues.push(item.value);
        sortedLabels.push(item.label);
    }

    return {
        sortedValues: sortedValues,//.map(v => ({value: v})),
        sortedLabels: sortedLabels
    };
}



function updateAxesAndChart() {
    updateAxes();
    displayChart();
}



function updateAxes() {
    var xLabels = [];
    var outputTicks = {};
    var fontSize = '20px';
    var xSvgWidth = '120px';
    var xSvgHeight = '50px';
    var ySvgWidth = '150px';
    var ySvgHeight = '150px';
    var ySvgFile = 'Cl.svg';
    var marginLeft = 80;
    var marginRight = 30;
    var yLabelMargin = 100;
    var yLabelOffset = -70;
    var lineWidth = 6;
    var stemWidth = 3;
    var pointSize = 4;
    var yMin = 0;
    var yMax = 8400;
    var yTicks = [0, 2000, 4000, 6000, 8000];
    var yLabels = ['0', '2000', '4000', '6000', '8000'];
    var yLabelsReduced = ['0', '4000', '8000'];
    if (selectedOption.value === "TE") {
	yMin = -230;
	yMax = 230;
	yTicks = [-200, -100, 0, 100, 200];
	yLabels = ['-200', '-100', '0', '100', '200'];
	yLabelsReduced = ['-200', '0', '200'];
    } else if (selectedOption.value === "EE") {
	yMin = -2;
	yMax = 60;
	yTicks = [0, 20, 40, 60];
	yLabels = ['0', '20', '40', '60'];
	yLabelsReduced = ['0', '60'];
    } else if (selectedOption.value === "PP") {
	yMin = 0;
	yMax = 2.5;
	yTicks = [0, 0.5, 1, 1.5, 2, 2.5];
	yLabels = ['0', '0.5', '1', '1.5', '2', '2.5'];
	yLabelsReduced = ['0', '1', '2'];
	ySvgFile = 'Cl_pp.svg';
	ySvgWidth = '260px';
	ySvgHeight = '260px';
	yLabelOffset = -140;
	yLabelMargin = 155;
    }
    if (document.getElementById('outputChart').offsetWidth > 500) {
	xLabels = ['10¹', '10²', "500", "1000", "1500", "2000", "2500"];
	outputTicks = generateXTicks(fewTicks = false);
    } else {
	xLabels = ['10¹', '10²', "1000", "2000"];
	outputTicks = generateXTicks(fewTicks = true);
	fontSize = '14px';
	xSvgWidth = '80px';
	xSvgHeight = '30px';
	ySvgWidth = '100px';
	ySvgHeight = '100px';
	marginLeft = 65;
	marginRight = 20;
	yLabelMargin = 65;
	yLabelOffset = -40;
	lineWidth = 3;
	stemWidth = 2;
	pointSize = 2.5;
	if (selectedOption.value === "PP") {
	    ySvgWidth = '170px';
	    ySvgHeight = '170px';
	    yLabelOffset = -80;
	    yLabelMargin = 100;
	}
    }
    xMajorTicks = outputTicks.majorTicks;
    xMinorTicks = outputTicks.minorTicks;
    const xMinorLabels = [];
    for (let i = 0; i < xMinorTicks.length; i++) {
        xMinorLabels.push("");
    }
    const { sortedValues: xTotalTicks, sortedLabels: xTotalLabels } = mergeSortAndSyncArrays(xMajorTicks, xLabels, xMinorTicks, xMinorLabels);

    if (document.getElementById('outputChart').offsetHeight < 400) {
	yLabels = yLabelsReduced;
	fontSize = '14px';
        xSvgWidth = '80px';
        xSvgHeight = '30px';
        ySvgWidth = '100px';
        ySvgHeight = '100px';
        marginLeft = 65;
        marginRight = 20;
        yLabelMargin = 65;
        yLabelOffset = -40;
	lineWidth = 3;
	stemWidth = 2;
	pointSize = 2.5;
	if (selectedOption.value === "PP") {
	    ySvgWidth = '170px';
	    ySvgHeight = '170px';
	    yLabelOffset = -80;
	    yLabelMargin = 100;
	}
    }

    
    const isDarkMode = document.body.classList.contains('dark-mode');
    const axisColor = isDarkMode ? 'white' : 'black';
    const labelFilter = isDarkMode ? "invert(100%) sepia(100%) saturate(0%) hue-rotate(201deg) brightness(106%) contrast(106%);" : "invert(0%) sepia(97%) saturate(0%) hue-rotate(34deg) brightness(89%) contrast(103%);";

    axisConfig.chart = {
        type: 'spline',
        backgroundColor: isDarkMode ? '#000000' : '#FFFFFF',
        animation: false, // Disable animation                                                                     
        marginLeft: marginLeft,
        marginRight: marginRight,
        marginTop: 15,
        plotBorderWidth: 2,
        plotBorderColor: axisColor,
    };
    axisConfig.title = {
	text: ''
    };
    axisConfig.xAxis = {
        type: 'linear',
        tickPositions: xTotalTicks,
        labels: {
            formatter: function() {
                return xTotalLabels[xTotalTicks.indexOf(this.value)] || '';
            },
            style: {
                color: axisColor,
                fontSize: fontSize
            },
            rotation: 0,
            overflow: 'allow',
        },
        title: {
            text: `<img src="./svg_images/ell.svg" style="width:${xSvgWidth}; height:${xSvgHeight}; filter:${labelFilter}"/>`,
            useHTML: true,
            align: 'middle',
        },
        lineColor: axisColor,
        lineWidth: 2,
        tickWidth: 2,
        tickLength: 10,
        tickColor: axisColor,
        min: 0,
        max: 1,
    };
    axisConfig.yAxis = {
        tickPositions: yTicks,
        labels: {
            formatter: function() {
                return yLabels.includes(this.value.toString()) ? this.value : '';
            },
            style: {
                color: axisColor,
                fontSize: fontSize
            },
            rotation: 270,
        },
        title: {
            text: `<img src="./svg_images/${ySvgFile}" style="position:absolute; width:${ySvgWidth}; height:${ySvgHeight}; filter:${labelFilter}"/>`,
            useHTML: true,
            rotation: 0,
            align: 'middle',
            margin: yLabelMargin,
            y: yLabelOffset,
        },
        gridLineWidth: 0,
        lineWidth: 2,
        lineColor: axisColor,
        tickWidth: 2,
        tickLength: 10,
        tickColor: axisColor,
        alignTicks: false,
        endOnTick: false,
        startOnTick: false,
        min: yMin,
        max: yMax,
    };
    axisConfig.series[0].lineWidth = lineWidth;
    axisConfig.series[2].stemWidth = stemWidth;
    axisConfig.series[1].marker.radius = pointSize;
};
    
async function updateDataAndChart() {
    // Get slider values
    const sliderValues = sliders.map(slider => parseFloat(slider.value));
    sliderValues.forEach((value, index) => {
        document.getElementById(`slider${index + 1}-value`).innerText = value;
    });

    // Create input tensor
    const inputData = tf.tensor([sliderValues]);
    const outputData = model.predict(inputData);
    
    // Extract the first 100 values
    const outputArray = await outputData.array();
    var slice0 = 0;
    var slice1 = 100;
    var factor = (10 ** 6 * 2.7255) ** 2;
    if (selectedOption.value === "TT") {
	slice0 = 0;
	slice1 = 100;
    } else if (selectedOption.value === "TE") {
	slice0 = 100;
	slice1 = 200;
    } else if (selectedOption.value === "EE") {
	slice0 = 200;
	slice1 = 300;
    } else if (selectedOption.value === "PP") {
	slice0 = 300;
	slice1 = 400;
	factor = 1e+7;
    }
    const spectrum = outputArray[0].slice(slice0, slice1).map(x => x * factor);

    const xdata = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21,23,25,27,30,33,36,40,44,49,54,60,67,75,83,92,103,115,128,143,160,179,200,223,249,278,311,348,387,426,465,504,543,582,621,660,699,738,777,816,855,894,933,972,1011,1050,1089,1128,1167,1206,1245,1284,1323,1362,1401,1440,1479,1518,1557,1596,1635,1674,1713,1752,1791,1830,1869,1908,1947,1986,2025,2064,2103,2142,2181,2220,2259,2298,2337,2376,2415,2454,2493]

    // Array af log-equidistant points from 2 to 2500
    const numPoints = 600;
    const startLog = Math.log10(2);
    const endLog = Math.log10(2500);
    const stepSize = (endLog - startLog) / (numPoints - 1);
    const xGrid = Array.from({ length: numPoints }, (_, index) => Math.pow(10, startLog + index * stepSize));
    // Interpolate the data
    const interpolatedData = interpolateData(spectrum, xdata, xGrid);

    if (data) {
	var seriesData = [];
	var errorData = [];
	if (selectedOption.value === "PP") {
	    seriesData = data.map(d => [scaleGraphPoint(d[0]), d[1] * 1e+7]);
	    errorData = data.map(d => [scaleGraphPoint(d[0]), d[1] * 1e+7 - d[2] * 1e+7, d[1] * 1e+7 + d[3] * 1e+7]);
	} else {
	    seriesData = data.map(d => [scaleGraphPoint(d[0]), d[1]]);
	    errorData = data.map(d => [scaleGraphPoint(d[0]), d[1] - d[2], d[1] + d[3]]);
	};
	axisConfig.series[1].data = seriesData;
	axisConfig.series[2].data = errorData;
    } else {
	axisConfig.series[1].data = [];
	axisConfig.series[2].data = [];
    };

    axisConfig.series[0].data = interpolatedData.map((y, i) => [scaleGraphPoint(xGrid[i]), y]);

    displayChart();
};

function displayChart() {
    Highcharts.setOptions({
	plotOptions: {
	    series: {
		animation: false,
		states: {
                    hover: {
			enabled: false
                    },
		    inactive: {
			enabled: false
		    }
		},
	    },
	}
    });
    Highcharts.chart('outputChart', axisConfig);
}


// Cubic spline interpolation function
function interpolateData(yData, xData, xGrid) {
    const n = yData.length;
    if (n < 2) {
        throw new Error('Data array must have at least 2 points for interpolation.');
    }

    // Prepare arrays for the spline interpolation
    const x = [...xData]; // Copy xData array
    const y = [...yData]; // Copy yData array

    // Calculate coefficients for the cubic spline
    const a = [];
    const b = [];
    const c = [];
    const d = [];
    const h = [];
    const alpha = [];
    const l = [];
    const mu = [];
    const z = [];

    // Step 1: Compute h[i] = x[i+1] - x[i]
    for (let i = 0; i < n - 1; i++) {
        h[i] = x[i + 1] - x[i];
    }

    // Step 2: Compute alpha[i] = (3/h[i]) * (y[i+1] - y[i]) - (3/h[i-1]) * (y[i] - y[i-1])
    for (let i = 1; i < n - 1; i++) {
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1]);
    }

    // Step 3: Compute l and mu
    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;

    for (let i = 1; i < n - 1; i++) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    // Step 4: Compute l[n], z[n], and c
    l[n - 1] = 1;
    z[n - 1] = 0;
    c[n - 1] = 0;

    for (let j = n - 2; j >= 0; j--) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
        a[j] = y[j];
    }

    // Step 5: Evaluate the spline at each desired point
    const interpolatedData = [];
    for (let i = 0; i < xGrid.length; i++) {
        const xValue = xGrid[i];
        let j = 0;
        // Find the segment to interpolate
        while (x[j] < xValue && j < n - 1) {
            j++;
        }
        if (j > 0) j--; // Ensure j is within bounds

        const dt = xValue - x[j];
        const interpolatedValue = a[j] + b[j] * dt + c[j] * dt ** 2 + d[j] * dt ** 3;
        interpolatedData.push(interpolatedValue);
    }

    return interpolatedData;
}
