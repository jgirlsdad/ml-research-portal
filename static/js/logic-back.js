// Create our map, giving it the streetmap and earthquakes layers to display on load
var currentYear = 0;      // hold the current year being displayed
var oldYear = 0;          // holds the previous year being displayed
window.plotLayer;   // previous leaflet layer that holds storm tracks
window.plotLayer2;  // previous leaflet layer that hold single storm track
window.trackLayer;  // leaflet layer that hold retired storm track
var mapTitle;           // title on leaflet map
var currentStorms;      // array of objects that holds all storms being displayed for the chosen year
var currentStorm;       // object that holds the current storm track being displayed 
var hoverLayer;         // leaflet layer that highlights the storm when you hover over the storm list
var maxStrength = new Object();  // stores the max stregths for the current list of storms

var height = 350

var variableSelected = ""
var citySelected = ""
var machineSelected = ""
var indicesSelectedString = ""

var phaseSelected = 0
var layerSelected = 1
var nodeSelected = 3
var modelPlot = 0
var numBins = 3
var classType = "CBIN"
var mlType    //  type of machine learning performed ... Regression or Classification

var tablesummary = window.open("template/summarytable.html", "Summary", "_blank")
tablesummary

var www_addr = "http://127.0.0.1:5000/"
//var www_addr = "https://demo-hurricanes-app.herokuapp.com/"

var cities = []
indices.forEach(function (data) {
  // console.log("indi", data)
})


var elem = document.querySelector('.grid');
var msnry = new Masonry(elem, {
  // options
  itemSelector: '.grid-item',
  columnWidth: 200,
  gutter: 20

});

// element argument can be a selector string
//   for an individual element
//var msnry = new Masonry( '.grid', {
// options
//});




function plotTS() {

  console.log("plotTS ", citySelected)
  d3.json(`${www_addr}getdata/${citySelected}`).then(function (data) {

    console.log("DATA  LEN ", Object.keys(data[0][variableSelected]).length)
    console.log("DD ", data[0][variableSelected])
    //  Object.keys(obj).length;

    y = Object.values(data[0][variableSelected])
    x = []
    xlabs = Object.keys(data[0][variableSelected])
    for (var nn = 0; nn <= Object.keys(data[0][variableSelected]).length; nn++) {
      x.push(nn)
    }
    console.log("XX TS", xlabs)
    console.log("YY TS", y)


    var trace1 = {
      x: xlabs,
      y: y,
      width: 400,
      height: 500,
      name: "Wind Speed",

      mode: 'scatter',
      marker: {
        size: 12,
        opacity: 1.0
      }
    };

    var selectorOptions = {
      buttons: [{
        step: 'year',
        stepmode: 'todate',
        count: 1,
        label: 'YTD'
      }, {
        step: 'year',
        stepmode: 'backward',
        count: 10,
        label: '10y'
      }, {
        step: 'all',
      }],
    };




    var layout = {
      xaxis: {
        rangeselector: selectorOptions,
        rangeslider: {}
      },
      title: `${citySelected} ${variableSelected}`,

    }





    var data = [trace1];
    Plotly.newPlot('plots', data, layout);

  })

}


var rectgrey
var rectcolor

function plotIndicies(map, indic) {
  lat1 = indic["lat1"]
  lat2 = indic["lat2"]
  lon1 = indic["lon1"]
  lon2 = indic["lon2"]

  var customOptions =
  {
    'maxWidth': '500',
    'className': 'custom',
    'permanent': true
  }


  // define rectangle geographical bounds
  var bounds = [[lat1, lon1], [lat2, lon2]];
  // create an orange rectangle
  rectgrey = L.rectangle(bounds, { color: "#A9A9A9", fillcolor: "#C0C0C0", weight: 1, opacity: 1. }).addTo(map);
  rectgrey.bindPopup("<b>Name:</b>" + indic['name'] + "<br>"
    + "<b>Acronym:</b>" + indic['code'] + "<br>"
    + "<b>URL:</b>" + indic['link'] + "<br>"
    + "<b>Description:</b> " + indic['desc']
    , customOptions)

  rectcolor = L.rectangle(bounds, { color: "#000000", fillcolor: "#000000", weight: 4, opacity: .25 })
  rectcolor.bindPopup("<b>Name:</b>" + indic['name'] + "<br>"
    + "<b>Acronym:</b>" + indic['code'] + "<br>"
    + "<b>URL:</b>" + indic['link'] + "<br>"
    + "<b>Description:</b> " + indic['desc']
    , customOptions)

  return

}



function plotIndicesTS(name) {

  console.log("plotIndicesTS ", name)



  //  Object.keys(obj).length;
  d3.json(`${www_addr}indices/${name}`).then(function (data2) {

    y = []

    xi = 0
    xlabs = []

    y = Object.values(data2[1]['ts'])

    x = Object.values(data2[0]['yearmo'])
    console.log(typeof (x))
    x.forEach(function (label) {
      console.log(label)
      temp = label.toString()
      hold = temp.substring(0, 4) + "-" + temp.substring(4, 6)
      xlabs.push(hold)
    })

    console.log("XXX ", xlabs)

    console.log("YYY ", y)

    var trace1 = {
      x: xlabs,
      y: y,
      width: 400,
      height: 500,
      name: "Wind Speed",

      mode: 'scatter',
      marker: {
        size: 12,
        opacity: 1.0
      }
    };


    var selectorOptions = {
      buttons: [{
        step: 'year',
        stepmode: 'todate',
        count: 1,
        label: 'YTD'
      }, {
        step: 'year',
        stepmode: 'backward',
        count: 10,
        label: '10y'
      }, {
        step: 'all',
      }],
    };


    var layout = {
      xaxis: {
        rangeselector: selectorOptions,
        rangeslider: {}
      },
      title: `${name}`,

    }

    var data = [trace1];
    Plotly.newPlot('indicesPlots', data, layout);


  })
}





function getData(name) {

  //  get storm data from flask server
  d3.json(`${www_addr}getdata/${name}`).then(function (cc) {


    plotTS(cc)
  })


}

//getData("NEW ORLEANS")


function getCities() {

  //  get storm data from flask server
  d3.json(`${www_addr}getcities`).then(function (cc) {
    console.log("AA  ", cc);
    cities = cc;
    plotCities()
    selectCity(cities)
  })


}

//  use this as a check point if things not working correctly
//retiredstorms()



function createMap() {
  // create the leaflet basemap
  var Map = L.map("map", {
    center: [
      0.00, 150.00
    ],
    zoom: 2,
    maxZoom: 6,
    minZoom: 1

  });
  return Map
}

function plotCities() {

  var locs = []
  // var map = createMap()
  var dots = []
  console.log("Plotting cities", cities)
  cities.forEach(function (data) {
    temp = []
    temp.push(parseFloat(data["lat"]))
    temp.push(parseFloat(data["lon"]))


    var dot = L.circle(temp, {
      color: "#ff0000",
      fillColor: "#ff0000",
      fillOpacity: .5,
      radius: 30000
    })
    dots.push(dot)
  })

  window.plotLayer2 = L.layerGroup(dots)
  myMap.addLayer(window.plotLayer2)

}


var myMap = createMap()
var cartodbAttribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="https://carto.com/attribution">CARTO</a>';

var positron = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', {
  attribution: cartodbAttribution
}).addTo(myMap);

getCities()
indices_layers = {}

indices.forEach(function (data) {

  plotIndicies(myMap, data)
  indices_layers[data.file] = {}
  indices_layers[data.file]['layercolor'] = rectcolor
  indices_layers[data.file]['state'] = 0
  indices_layers[data.file]['layergrey'] = rectgrey

})



function selectVar() {
  variables = [
    {
      "label": "Average Temperature",
      "abbr": "TAVG"
    },
    {
      "label": "Minimum Temperature",
      "abbr": "TMIN"
    },
    {
      "label": "Maximum Temperature",
      "abbr": "TMAX"
    },
    {
      "label": "Precipitation",
      "abbr": "PRCP",
    }
  ]
  var form = d3.select("#variables").append("form").attr("id", "varform")



  labels = (form.selectAll("label")
    .data(variables)
    .enter()
    .append("label")
    .attr('id', function (d) {

      return d.abbr
    })
    .attr('class', 'labels')
    .text(function (d) {
      return d.label
    }))
    .append("input")
    .attr('id', 'varselect')
    .attr('type', 'radio')
    .attr('name', 'mode')
    .attr('value', function (d) {
      return d.abbr;
    })
    .on('change', checkVariable)
}

function selectCity(cities) {

  // var form = d3.select("#city").append("form").attr("id","cityform")


  var selector = d3.select("#city")
    .append("select")
    .attr("id", "cityselect")
    .on("change", checkCity)
    .selectAll("option")
    .data(cities)
    .enter().append("option")
    .text(function (d, i) {
      return d.name
    })
    .attr("value", function (d, i) {
      return d.name;
    })

}

function checkCity() {
  citySelected = d3.select(this).property('value')

  checkInput()
}

function checkVariable() {
  variableSelected = d3.select(this).property('value')
  console.log("VARIABLE SEL", variableSelected)
  checkInput()
}

function checkInput() {


  if (variableSelected && citySelected) {
    console.log("GOT BOTH ", variableSelected, citySelected)
    plotTS()
  } else if (variableSelected) {
    console.log("NO CITY ", variableSelected)
  } else if (citySelected) {
    console.log("NO VARIABLE ", citySelected)
  }


}


function selectIndices() {

  var form = d3.select("#indices").append("form").attr("id", "indform")

  // console.log("VARS ", variables)

  var label_divs = form.selectAll(".checkboxes")
    .data(indices)
    .enter()
    // append a div
    .append('div')
    .attr('class', 'checkbox')
  // add an event handler
  // append an input
  // append a label
  var label_divs_label = label_divs.append("label");

  label_divs_label.append("input")
    .attr('id', d => d.file)
    .attr('type', 'checkbox')
    .attr('name', function (d) {
      console.log("CB ", d.file)
      return d.code;
    })
    .attr('value', function (d) {
      return d.file;
    })
    .on('change', function () {
      var checked = []
      indicesSelectedString = ""
      console.log("going to plotindicests")
      plotIndicesTS(this.value)
      // var boxes = d3.selectAll("input.checkbox");
      var boxes = d3.selectAll("input[type='checkbox']:checked");
      boxes.each(function () {
        checked.push(this.value)
        indicesSelectedString = indicesSelectedString + "," + this.value
        console.log("VAL ", this.value)
      });

      keys = Object.keys(indices_layers)
      console.log("checked  ", checked)
      keys.forEach(function (data) {
        if (checked.includes(data)) {
          myMap.removeLayer(indices_layers[data]['layergrey'])
          myMap.addLayer(indices_layers[data]['layercolor'])
          indices_layers[data]['state'] = 1
        } else {
          if (indices_layers[data]['state'] == 1) {
            myMap.removeLayer(indices_layers[data]['layercolor'])
            myMap.addLayer(indices_layers[data]['layergrey'])
            indices_layers[data]['state'] = 0
          }
        }
      })

      console.log("CHECKED ", checked)
      indicesSelectedString = indicesSelectedString.substring(1)
    });

  label_divs_label.append('span')
    // .attr('id', function (d) {
    //   return d.file
    // })
    .attr('for', function (d) {
      return d.file
    })
    .text(function (d) {
      return d.name
    })
    .attr('class', 'labels')

}


function checkInd() {
  indicesSelectedString = ""
  indicesSelected = d3.selectAll("input.checkbox:checked");
  indicesSelected.forEach(function () {
    indicesSelectedString = indicesSelectedString + "," + this.value

  })
}



function selectMachine() {
  var variables = [
    {
      "label": "Regression",
      "abbr": "regrlinear"
    }
  ]
  var form = d3.select("#regression").append("form").attr("id", "varform")



  labels = (form.selectAll("label")
    .data(variables)
    .enter()
    .append("label")
    .attr('id', function (d) {

      return d.abbr
    })
    .attr('class', 'labels')
    .text(function (d) {
      return d.label
    }))
    .append("input")
    .attr('id', 'mlselect')
    .attr('type', 'radio')
    .attr('name', 'mode')
    .attr('value', function (d) {
      return d.abbr;
    })
    .on('change', checkMachine)
}


function checkMachine() {
  machineSelected = d3.select(this).property('value')


  if (variableSelected && citySelected && indicesSelectedString) {

    machineLearn()
  } else if (variableSelected) {

  } else if (citySelected) {

  }


}

function checkInputsAll(what) {
  // machineSelected = d3.select(this).property('value')
  console.log("WHAT ", what)
  machineSelected = 1
  if (variableSelected && citySelected && indicesSelectedString) {
    if (what == 1) {
      machineLearn()
    } else if (what == 2) {
      category()
    }
  } else {
    if (!citySelected) {
      alert("Please select a <b>city</b> to process at the top of the page then hit the Machine Learn button again")
    } else if (!variableSelected) {
      alert("Please select a <b>Variable</b> at the top of the page then hit the Machine Learn button again")
    } else if (!indicesSelectedString) {
      alert("Please select two or more  <b>Climate Indices</b>  then hit the Machine Learn button again")
    }
  }
}

function plotModels(ts, phases, type, model, which) {

  //  countrySelector.value
  
  var myPlot = document.getElementById('modelPlot'),
    hoverInfo = document.getElementById('modelPlot')
  if (which == 1) { //  regression
    models = ['SVR', 'Linear', 'Bayesian']
    mtype = "Regression"
  } else if (which == 2) {  // categories
    models = ['Logistic', 'SVC', 'RFC']
    mtype = "Classify"
  }
  console.log("PLOTMODEL",phases,type,model,which,ts)

  if (type) {
    y1 = ts[models[0]][type]
    y2 = ts[models[1]][type]
    y3 = ts[models[2]][type]
    title = `${mtype} - ${type} for all models`
    name1 = models[0]
    name2 = models[1]
    name3 = models[2]
  } else if (model) {
    y1 = ts[model]['r2']
    y2 = ts[model]['test_score']

    y3 = ts[model]['train_score']
    title = `${mtype} - ${model} r2,test and train scores`
    name1 = 'R2'
    name2 = 'Test'
    name3 = 'Train'

  }

  xlabs = phases

  
  //  trace1,trace2,trace3 =  getTraces(phases,ts['SVR']['r2'],ts['linear']['r2'],ts['bayesian']['r2'])

  var trace1 = {
    x: xlabs,
    y: y1,
    width: 400,
    height: height,
    name: name1,

    mode: 'lines and markers',
    marker: {
      size: 12,
      opacity: 1.0
    }

  }

  var trace2 = {
    x: xlabs,
    y: y2,
    width: 400,
    height: height,
    name: name2,
    mode: 'lines and markers',
    marker: {
      size: 12,
      opacity: 1.0
    }


  };

  var trace3 = {
    x: xlabs,
    y: y3,
    width: 400,
    height: height,
    name: name3,

    mode: 'lines and markers',
    marker: {
      size: 12,
      opacity: 1.0
    }

  };

  var layout = {
    title: title,
    hovermode: 'closest',
    width: 400,
    height: height,
    yaxis: {
      title: "Score"
    },
    xaxis: {
      title: "Phase Shift (months)"
    }
  }

  var data = [trace1, trace2, trace3];
  Plotly.newPlot('modelPlot', data, layout);
  modelPlot = 1

  myPlot.on('plotly_click', function (data) {

    var infotext = data.points.map(function (d) {
      ncurve = d.curveNumber
      return (d.pointNumber);
    });
    
    nphase = infotext[0]
    
    y_obs = obs[models[ncurve]][nphase]['obs']
    y_pred = obs[models[ncurve]][nphase]['pred']
    plotPred(y_obs, y_pred, models[ncurve], nphase, "plots", 1)

   

    //    hoverInfo.innerHTML = infotext.join('<br/>');

  })



}


function plotPred(y1, y2, model, phase, div, ptype) {
  console.log("DIV DIV ", y1)
  console.log("YYYY2 ", y2)
  var xx = []
  ylength = y1.length
  for (nn = 0; nn <= ylength; nn += 1) {
    xx.push(nn)

  }
  if (ptype == 1) {
    title = `Obs and Pred for ${model} with phase shift of ${phase} months`
    name1 = 'Observed'
    name2 = 'Predicted'
    ytitle = 'Value'
    xtitle = "Time"
  } else {
    title = `Train and Test MSE erros  for ${model} with phase shift of ${phase} months`
    name1 = 'MSE'
    name2 = 'MSE_val'
    ytitle = "mse"
    xtitle = "Epochs"
  }
  var trace1 = {
    x: xx,
    y: y1,
    width: 400,
    height: height,
    name: name1,

    mode: 'lines and markers',
    marker: {
      size: 12,
      opacity: 1.0
    }

  }

  var trace2 = {
    x: xx,
    y: y2,
    width: 400,
    height: height,
    name: name2,
    mode: 'lines and markers',
    marker: {
      size: 12,
      opacity: 1.0
    }


  };


  var layout = {
    title: title,
    hovermode: 'closest',
    yaxis: {
      title: ytitle
    },
    xaxis: {
      title: xtitle
    }
  }


  if (modelPlot) {
    Plotly.deleteTraces(div, 0)
  }
  //  console.log("# OF TRACES ",graphDiv.data.length)
  var data = [trace1, trace2];
  Plotly.newPlot(div, data, layout);
}




function machineLearn() {


  d3.json(`${www_addr}machine/${citySelected}/${variableSelected}/${machineSelected}/${indicesSelectedString}`).then(function (scores) {
    obs = {}
    obs['SVR'] = {}
    obs['Linear'] = {}
    obs['Bayesian'] = {}
    modelsR = ['SVR', 'Linear', 'Bayesian']
    modelsC = ['Logistic', 'SVC', 'RFC']
    stats = {}
    modelsR.forEach(function (model) {
      stats[model] = {}
      stats[model]['R2'] = -2
      stats[model]['Phase'] = -2
    })
    stats['Neural'] = {}
    stats['Neural']['R2'] = " "

    modelsC.forEach(function (model) {
      stats[model] = {}
      stats[model]['R2'] = " "
      stats[model]['Phase'] = " "
    })




    ts = {}
    scores.forEach(function (data) {
      r2 = []
      test_score = []
      train_score = []

      phases = []

      data.forEach(function (temp) {

        phase_shift = Object.keys(temp)

        phase_shift.forEach(function (d) {
          console.log("phases ", d)
          phase = d
          phases.push(d)
        })
        console.log("TEMP ", temp)

        vals = Object.values(temp)
        console.log("Vals ", vals)
        r2.push(vals['0']['r2'])
        test_score.push(vals['0']['test'])
        train_score.push(vals['0']['train'])
        model = vals['0']['model']
        rhold = vals['0']['r2'].toFixed(2)
        if (rhold > stats[model]['R2']) {
          stats[model]['R2'] = rhold
          stats[model]['Phase'] = phase
        }

        obs[model][phase] = {}
        obs[model][phase]['obs'] = vals[0]['y_obs']
        obs[model][phase]['pred'] = vals[0]['y_pred']


        //    data[model][phase]['obs'] = vals['0']['y_obs']
        //     obs[model][phase]['obs'] = vals['0']['y_obs']
        console.log("DATA ", obs)

      })
      ts[model] = []
      ts[model]['r2'] = r2
      ts[model]['test_score'] = test_score
      ts[model]['train_score'] = train_score
    })
    console.log("TS ", obs)
    plotModels(ts, phases, "r2", "", 1)


    updateTableC(modelsR, modelsC, stats, "R")


    choices = ['r2', 'test_score', 'train_score']
   

    selectTS(choices,ts,phases,1)
    selectMod(modelsR,ts,phases,1)
  })  
}

function selectPhaseold() {

  var phases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]



  var models = [
    {
      "model": "SVR",
    },
    {
      "model": "Linear",
    },
    {
      "model": "Bayesian",
    },
  ]
  var form = d3.select("#categories").append("form").attr("id", "catform")


  labels = (form.selectAll("label")
    .data(phases)
    .enter()
    .append("label")
    .attr('id', function (d) {
      console.log("var name", d)
      return d
    })
    .attr('class', 'labels')
    .text(function (d) {
      return d
    }))
    .append("input")
    .attr('id', 'varselect')
    .attr('type', 'radio')
    .attr('name', 'mode')
    .attr('value', function (d) {
      return d;
    })

}


function selectPhases() {

  // var form = d3.select("#city").append("form").attr("id","cityform")
  var phases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


  var selector = d3.select("#phases")
    .append("select")
    .attr("id", "phaseselect")
    .on("change", setPhase)
    .selectAll("option")
    .data(phases)
    .enter().append("option")
    .text(function (d, i) {
      return d
    })
    .attr("value", function (d, i) {
      return d;
    })


}

function setPhase() {
  phaseSelected = d3.select(this).property('value')
  console.log("PHASE SEL", phaseSelected)
}



function selectLayers() {

  // var form = d3.select("#city").append("form").attr("id","cityform")
  var Layers = [1, 2, 3, 4]


  var selector = d3.select("#layers")
    .append("select")
    .attr("id", "layerelect")
    .on("change", setLayer)
    .selectAll("option")
    .data(Layers)
    .enter().append("option")
    .text(function (d) {
      return d
    })
    .attr("value", function (d) {
      return d;
    })

}

function setLayer() {
  layerSelected = d3.select(this).property('value')

}



function selectNodes() {

  // var form = d3.select("#city").append("form").attr("id","cityform")
  var Nodes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


  var selector = d3.select("#nodes")
    .append("select")
    .attr("id", "layerelect")
    .on("change", setNode)
    .selectAll("option")
    .data(Nodes)
    .enter().append("option")
    .text(function (d) {
      return d
    })
    .attr("value", function (d) {
      return d;
    })

}

function setNode() {
  nodeSelected = d3.select(this).property('value')

}




function neural() {


  d3.json(`${www_addr}neural/${citySelected}/${variableSelected}/${indicesSelectedString}/${phaseSelected}/${layerSelected}/${nodeSelected}`).then(function (scores) {

    models = ['Neural', 'SVR', 'Linear', 'Bayesian']
    stats = {}
    models.forEach(function (model) {
      stats[model] = {}
      stats[model]['R2'] = " "
      stats[model]['Phase'] = " "
    })


    scores.forEach(function (data) {


      console.log("Data ", data)
      r2 = parseFloat(data['r2'])
      stats['Neural']['R2'] = r2.toFixed(2)
      y_obs = data['y_obs']
      y_pred = data['y_pred']
      acc = data['mse']
      val_acc = data['mse_val']
      console.log("ACC ", acc)
      console.log("VACC ", val_acc)
      plotPred(y_obs, y_pred, "Neural Network", phaseSelected, "plots", 1)
      plotPred(acc, val_acc, "Neural Network", phaseSelected, "modelPlot", 2)
      modelPlot = 1

      //        obs[model][phase] = {}
      //        obs[model][phase]['obs'] = vals[0]['y_obs']
      //        obs[model][phase]['pred'] = vals[0]['y_pred']


      //    data[model][phase]['obs'] = vals['0']['y_obs']
      //     obs[model][phase]['obs'] = vals['0']['y_obs']


    })
    //  console.log("TS ", obs)
    //   plotModels(ts, phases, "r2", "")


    updateTable(stats)

    function updateTable(stat) {
      table = document.querySelector("table")
      row = table.insertRow()
      cell = row.insertCell()
      text = document.createTextNode(citySelected);
      cell.appendChild(text)
      cell = row.insertCell()
      text = document.createTextNode(indicesSelectedString);
      cell.appendChild(text)
      cell = row.insertCell()
      text = document.createTextNode(variableSelected);
      cell.appendChild(text)
      cell = row.insertCell()
      text = document.createTextNode(stats['Neural']['R2']);
      cell.appendChild(text)
    }
    cell = row.insertCell()
    text = document.createTextNode(phaseSelected);
    cell.appendChild(text)
    cell = row.insertCell()
    text = document.createTextNode(`${layerSelected}/${nodeSelected}`);
    cell.appendChild(text)


  })


}




function category() {


  d3.json(`${www_addr}classify/${citySelected}/${variableSelected}/${indicesSelectedString}/${classType}/${numBins}`).then(function (scores) {
    //  d3.json(`${www_addr}classify/NEW ORLEANS/TAVG/nina1,nina4`).then(function (scores) {
    console.log("IN category")


    modelsR = ['SVR', 'Linear', 'Bayesian']
    modelsC = ['Logistic', 'SVC', 'RFC']

    obs = {}


    stats = {}
    modelsR.forEach(function (model) {
      stats[model] = {}
      stats[model]['R2'] = ' '
      stats[model]['Phase'] = ' '
      obs[model] = {}
    })




    modelsC.forEach(function (model) {
      stats[model] = {}
      stats[model]['R2'] = -2
      stats[model]['Phase'] = -2
      obs[model] = {}
    })
    stats['Neural'] = {}
    stats['Neural']['R2'] = " "
    ts = {}
    scores.forEach(function (data) {
      r2 = []
      test_score = []
      train_score = []

      phases = []

      data.forEach(function (temp) {

        phase_shift = Object.keys(temp)

        phase_shift.forEach(function (d) {
          console.log("phases ", d)
          phase = d
          phases.push(d)
        })
        console.log("TEMP ", temp)

        vals = Object.values(temp)
        console.log("Vals ", vals)
        r2.push(vals['0']['r2'])
        test_score.push(vals['0']['test'])
        train_score.push(vals['0']['train'])
        model = vals['0']['model']
        rhold = vals['0']['r2'].toFixed(2)
        if (rhold > stats[model]['R2']) {
          stats[model]['R2'] = rhold
          stats[model]['Phase'] = phase
        }

        obs[model][phase] = {}
        obs[model][phase]['obs'] = vals[0]['y_obs']
        obs[model][phase]['pred'] = vals[0]['y_pred']


        //    data[model][phase]['obs'] = vals['0']['y_obs']
        //     obs[model][phase]['obs'] = vals['0']['y_obs']
        console.log("DATA ", obs)

      })
      ts[model] = []
      ts[model]['r2'] = r2
      ts[model]['test_score'] = test_score
      ts[model]['train_score'] = train_score
    })
    console.log("TS ", ts)
    plotModels(ts, phases, "r2", "", 2)


    updateTableC(modelsR, modelsC, stats, "C")



    var innerContainer = document.querySelector('#output'),
      // plotEl = innerContainer.querySelector('#varSelect'),
      variableSelector = document.querySelector('.varSelect');
//    modelSelector = document.querySelector('.modelSelect');



    choices = ['r2', 'test_score', 'train_score']

 
//    assignOptions(choices, modelSelector);
    selectTS(choices,ts,phases,2)
    selectMod(modelsC,ts,phases,2)
 //   variableSelector.addEventListener('change', updateplotModels, false);
 //   modelSelector.addEventListener('change', updateplotModels2, false);


    



  })

}



function selectMod(models,ts,phases,which) {

  // var form = d3.select("#city").append("form").attr("id","cityform")

  d3.select("#modelselect").remove()
  var modelSelector = d3.select("#modelSelect")
    .append("select")
    .attr("id", "modelselect")
    .on("change", function (d) {
      console.log("CHANGE",this.value,"   ",d)
      plotModels(ts, phases, "",this.value,which);
    })
    .selectAll("option")
    .data(models)
    .enter().append("option")
    .text(function (d, i) {
      return d
    })
    .attr("value", function (d, i) {
      console.log("SELECT MODEL ",d)
      return d;
    })

}



function selectTS(choices,ts,phases,which) {

  // var form = d3.select("#city").append("form").attr("id","cityform")

  d3.select("#tsselect").remove()
  var modelSelector = d3.select("#tsSelect")
    .append("select")
    .attr("id", "tsselect")
    .on("change", function (d) {
      console.log("CHANGE",this.value,"   ",d)
      plotModels(ts, phases,this.value,"",which);
    })
    .selectAll("option")
    .data(choices)
    .enter().append("option")
    .text(function (d, i) {
      return d
    })
    .attr("value", function (d, i) {
      console.log("SELECT MODEL ",d)
      return d;
    })

}



function updateplotModels(ts,phases,which) {
  type = this.value
  console.log("UPDATING MODELS ",type,phases,ts)
  if (modelPlot) {
    //      Plotly.deleteTraces('output', 0)
    Plotly.deleteTraces('modelPlot', 0)
  }
  plotModels(ts, phases, type, "",which);
}


function updateplotModels2(ts,phases,which) {
  model = this.value
  if (modelPlot) {
    Plotly.deleteTraces("modelPlot", 0)
    //      Plotly.deleteTraces('output', 0)

  }
  plotModels(ts, phases, "", model,which);
}



function updateTableC(modelsr, modelsc, stats, mlType) {
  table = document.querySelector("table")
  row = table.insertRow()
  cell = row.insertCell()
  text = document.createTextNode(citySelected);
  cell.appendChild(text)
  cell = row.insertCell()
  text = document.createTextNode(indicesSelectedString);
  cell.appendChild(text)


  cell = row.insertCell()
  text = document.createTextNode(variableSelected);
  cell.appendChild(text)

  cell = row.insertCell()
  text = document.createTextNode(stats['Neural']['R2']);
  cell.appendChild(text)

  cell = row.insertCell()
  text = document.createTextNode(" ");
  cell.appendChild(text)
  cell = row.insertCell()
  text = document.createTextNode(" ");
  cell.appendChild(text)

  modelsr.forEach(function (modl) {
    cell = row.insertCell()
    text = document.createTextNode(stats[modl]['R2']);
    cell.appendChild(text)

    cell = row.insertCell()
    text = document.createTextNode(stats[modl]['Phase']);
    cell.appendChild(text)
  })

  if (mlType == "R") {
    return
  }

  cell = row.insertCell()
  text = document.createTextNode(classType);
  cell.appendChild(text)




  if (classType == "PBIN") {
    mm = 7
  } else {
    mm = numBins
  }
  cell = row.insertCell()
  text = document.createTextNode(mm);
  cell.appendChild(text)


  modelsc.forEach(function (modl) {
    cell = row.insertCell()
    text = document.createTextNode(stats[modl]['R2']);
    cell.appendChild(text)

    cell = row.insertCell()
    text = document.createTextNode(stats[modl]['Phase']);
    cell.appendChild(text)
  })
}




function selectBins() {
  variables = [
    {
      "label": "Provide STDDEV Bins",
      "abbr": "PBIN"
    },
    {
      "label": "Create Bins",
      "abbr": "CBIN"
    }
  ]
  var form = d3.select("#catopts").append("form").attr("id", "optform")



  labels = (form.selectAll("label")
    .data(variables)
    .enter()
    .append("label")
    .attr('id', function (d) {

      return d.abbr
    })
    .attr('class', 'labels')
    .text(function (d) {
      return d.label
    }))
    .append("input")
    .attr('id', 'catselect')
    .attr('type', 'radio')
    .attr('name', 'mode')
    .attr('value', function (d) {
      return d.abbr;
    })
    .attr('checked', function (d, i) {
      console.log("RADFIO ", i)
      if (i == 0) {
        console.log("False")
        return 0

      } else {
        console.log("Ture")
        return 0
      }
    })
    .on('change', checkCatOpts)

  data = [3, 4, 5, 6, 7, 8, 9, 10]
  var sliderStep = d3
    .sliderBottom()
    .min(d3.min(data))
    .max(d3.max(data))
    .width(100)
    .tickFormat(d3.format("2.2"))
    .tickValues(data)
    .step(1)
    .default(3)
    .on('onchange', val => {
      numBins = val
      return val;
    });

  var gStep = d3
    .select('#catslider')
    .append('svg')
    .attr('width', 150)
    .attr('height', 70)
    .attr("visibility", "visible")
    .append('g')
    .attr('transform', 'translate(30,30)')
    .call(sliderStep)
  // gStep.call(sliderStep);
  // d3.select('#catslider').call(slider);

}

function checkCatOpts() {
  //  if (this.value == "PBIN") {
  //    console.log("I made it",this.value)
  //    d3.select("#catslider").attr("visibility","hidden")
  //    numBins=7
  //  }
  classType = this.value
}



category()

selectNodes()
selectLayers()
selectPhases()
selectVar()
selectIndices()
selectBins()
//selectMachine()