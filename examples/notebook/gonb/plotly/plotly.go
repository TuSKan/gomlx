// Package plotly uses GoNB plotly support (`github.com/janpfeifer/gonb/gonbui/plotly`) to plot both
// on dynamic plots while training or to quickly plot the results of a previously
// saved plot results in a checkpoints directory.
//
// In either case it allows adding baseline plots of previous checkpoints.
//
// The advantage of `plotly` over `margaid` plots is that it uses Javascript to make the plot interactive (it displays
// information on mouse hover).
//
// The disadvantage is that saving doesn't work, because of the javascript nature.
package plotly

import (
	"fmt"
	grob "github.com/MetalBlueberry/go-plotly/graph_objects"
	"github.com/gomlx/gomlx/examples/notebook/gonb/plots"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/janpfeifer/gonb/gonbui"
	"github.com/janpfeifer/gonb/gonbui/dom"
	gonbplotly "github.com/janpfeifer/gonb/gonbui/plotly"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"math"
	"os"
	"path"
)

// PlotConfig hold the configuration object that will generate the plot. Create it with [New].
type PlotConfig struct {
	// figs is the list of figures, one per metric type.
	figs []*grob.Fig

	// metricsNames maps the metric name to the trace in the corresponding figure.
	// One per figure.
	metricsNamesToTrace []map[string]int

	// metricsTypesToFig maps a metric type to a figure index in `figs`.
	metricsTypesToFig map[string]int

	// pointsAdded limits plotting only if enough points have been added.
	pointsAdded int

	// EvalDatasets registered to be used during evaluation when dynamically capturing points during training.
	EvalDatasets []train.Dataset

	// gonbId of the `<div>` tag where to generate dynamic plots.
	gonbId string

	// finalPlot indicates whether the final plot has already been drawn.
	finalPlot bool

	// filePath where to save data points to. Only used if not empty.
	enablePointsWriting bool
	filePath            string
	fileWriter          chan<- plots.Point
	errFileWriter       <-chan error
}

// New creates a new PlotConfig, that can be used to generate plots.
func New() *PlotConfig {
	return &PlotConfig{
		metricsTypesToFig: make(map[string]int),
	}
}

// Dynamic sets plot to be dynamically updated and new data comes in. It's a no-op if not running in a GoNB notebook.
//
// `datasets` is a list of datasets to be evaluated when collecting metrics for plotting.
//
// It should be followed by a call to [ScheduleExponential] or [SechedulePeriodic] (or both) to schedule capturing
// points to plot, and [WithCheckpoint] to save the captured points.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) Dynamic(datasets ...train.Dataset) *PlotConfig {
	if !gonbui.IsNotebook {
		return pc
	}
	pc.gonbId = gonbui.UniqueId()
	pc.EvalDatasets = datasets
	if pc.pointsAdded < 3 {
		// If we are having a dynamically updating plot, we reserve the transient HTML block
		// upfront -- otherwise it will interfere with the progressbar the first time it is displayed.
		gonbui.UpdateHtml(pc.gonbId, "(...collecting metrics, minimum 3 required to start plotting...)")
	} else {
		gonbui.UpdateHtml(pc.gonbId, "")
		pc.DynamicPlot(false)
	}
	return pc
}

// ScheduleExponential collection of plot points, starting at `startStep` and with an increasing step factor
// of `stepFactor`. Typical values where could be 100 and 1.1.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) ScheduleExponential(loop *train.Loop, startStep int, stepFactor float64) *PlotConfig {
	train.ExponentialCallback(loop, startStep, stepFactor, true,
		"plotly.DynamicPlot", 0, func(loop *train.Loop, metrics []tensor.Tensor) error {
			// Update plots with metrics.
			return plots.AddTrainAndEvalMetrics(pc, loop, metrics, pc.EvalDatasets)
		})
	pc.attachOnEnd(loop)
	return pc
}

// ScheduleNTimes of collection of plot points.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) ScheduleNTimes(loop *train.Loop, numPoints int) *PlotConfig {
	train.NTimesDuringLoop(loop, numPoints, "plotly.DynamicPlot", 0,
		func(loop *train.Loop, metrics []tensor.Tensor) error {
			return plots.AddTrainAndEvalMetrics(pc, loop, metrics, pc.EvalDatasets)
		})
	pc.attachOnEnd(loop)
	return pc
}

// attachOnEnd registers a final call to DynamicPlot when training finishes. After that no more dynamic plots
// are allowed.
func (pc *PlotConfig) attachOnEnd(loop *train.Loop) {
	loop.OnEnd("margaid plots", 120, func(_ *train.Loop, _ []tensor.Tensor) error {
		// Final plot: only called once to the transient plots
		if pc.gonbId != "" && !pc.finalPlot {
			// Erase intermediary transient plots.
			pc.DynamicPlot(true)
			pc.finalPlot = true
		}
		pc.stopWriting()
		return nil
	})
}

// WithCheckpoint uses the `checkpointDir` both to load data points and to save any new data points.
// Usually, used with [PlotConfig.Dynamic].
//
// New data-points are saved asynchronously -- not to slow down training, with the downside of
// potentially having I/O issues reported asynchronously.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) WithCheckpoint(checkpointDir string) *PlotConfig {
	// Ignore errors while loading: maybe nothing was written yet.
	_ = pc.LoadCheckpointData(checkpointDir)
	checkpointDir = data.ReplaceTildeInDir(checkpointDir)
	filePath := path.Join(checkpointDir, plots.TrainingPlotFileName)
	pc.fileWriter, pc.errFileWriter = plots.CreatePointsWriter(filePath)
	pc.enablePointsWriting = true
	return pc
}

// stopWriting indicates that no more points are coming. This closes the asynchronous job writing new points.
func (pc *PlotConfig) stopWriting() {
	if pc.fileWriter != nil {
		close(pc.fileWriter)
		pc.fileWriter = nil
		err := <-pc.errFileWriter
		if err != nil {
			klog.Errorf("Failed to write plots data: %+v", err)
		}
		pc.enablePointsWriting = false
	}
}

// PointFilter can change any [plots.Point] arbitrarily. If it returns false means the point should be dropped.
type PointFilter func(p *plots.Point) bool

// LoadCheckpointData loads plotting data from a checkpoint path.
// Notice this only works if the model was trained with plotting, with the metrics saved into the file
// `training_plot_points.json` ([plots.TrainingPlotFileName]).
// Notice that if `dataDirOrFile` is a file it reads from that instead.
//
// The `filters` are an optional list of filters to apply (in order) to each of the points read: it allows points
// to be modified arbitrarily -- in particular useful to change names (like adding a prefix) of metrics or metrics
// types.
//
// Each filter can also eliminate points by returning false -- only points for which filters returned true are
// included.
func (pc *PlotConfig) LoadCheckpointData(dataDirOrFile string, filters ...PointFilter) error {
	dataDirOrFile = data.ReplaceTildeInDir(dataDirOrFile)
	fi, err := os.Stat(dataDirOrFile)
	if err != nil {
		return errors.Wrapf(err, "plotly.LoadCheckpointData(%q) cannot stat the file", dataDirOrFile)
	}
	var points []plots.Point
	if fi.IsDir() {
		points, err = plots.LoadPointsFromCheckpoint(dataDirOrFile)
	} else {
		points, err = plots.LoadPoints(dataDirOrFile)
	}
	if err != nil {
		return errors.WithMessagef(err, "plotly.LoadCheckpointData(%q) failed to load points", dataDirOrFile)
	}

	steps := types.MakeSet[float64]()
	enableWriting := pc.enablePointsWriting
	pc.enablePointsWriting = false
	defer func() { pc.enablePointsWriting = enableWriting }()

nextPoint:
	for _, point := range points {
		for _, filter := range filters {
			if !filter(&point) {
				continue nextPoint
			}
		}
		pc.AddPoint(point)
		if !steps.Has(point.Step) {
			pc.pointsAdded++ // Count number of points per different value of global step.
			steps.Insert(point.Step)
		}
	}
	return nil
}

// AddPoint add one point to the plots.
//
// Usually not called directly, instead use [LoadCheckpointData] or [Dynamic], which will attach to a
// training loop and call this automatically.
func (pc *PlotConfig) AddPoint(pt plots.Point) {
	if math.IsNaN(pt.Value) || math.IsInf(pt.Value, 0) || math.IsNaN(pt.Step) || math.IsInf(pt.Step, 0) {
		// Ignore invalid points.
		return
	}
	if pc.fileWriter != nil {
		// Save point asynchronously.
		pc.fileWriter <- pt
	}
	figIdx, found := pc.metricsTypesToFig[pt.MetricType]
	if !found {
		pc.figs = append(pc.figs, &grob.Fig{
			Layout: &grob.Layout{
				Title: &grob.LayoutTitle{
					Text: pt.MetricType,
				},
				Xaxis: &grob.LayoutXaxis{
					Showgrid: grob.True,
					Type:     grob.LayoutXaxisTypeLog,
				},
				Yaxis: &grob.LayoutYaxis{
					Showgrid: grob.True,
					Type:     grob.LayoutYaxisTypeLog,
				},
				Legend: &grob.LayoutLegend{
					//Y:       -0.2,
					//X:       1.0,
					//Xanchor: grob.LayoutLegendXanchorRight,
					//Yanchor: grob.LayoutLegendYanchorTop,
				},
			},
		})
		pc.metricsNamesToTrace = append(pc.metricsNamesToTrace, make(map[string]int))
		figIdx = len(pc.figs) - 1
		pc.metricsTypesToFig[pt.MetricType] = figIdx
	}
	fig := pc.figs[figIdx]
	metricNameToTrace := pc.metricsNamesToTrace[figIdx]

	traceIdx, found := metricNameToTrace[pt.MetricName]
	if !found {
		metricNameToTrace[pt.MetricName] = len(fig.Data)
		traceIdx = len(fig.Data)
		fig.Data = append(fig.Data, &grob.Scatter{
			Name: pt.MetricName,
			Type: grob.TraceTypeScatter,
			Line: &grob.ScatterLine{
				Shape: grob.ScatterLineShapeLinear,
			},
			Mode: grob.ScatterMode("lines+markers"),
			X:    []float64{},
			Y:    []float64{},
		})
	}
	trace := fig.Data[traceIdx].(*grob.Scatter)
	trace.X = append(trace.X.([]float64), pt.Step)
	trace.Y = append(trace.Y.([]float64), pt.Value)
}

// DynamicSampleDone is called after all the data points recorded for this sample (evaluation at a time step).
// The value `incomplete` is set to true if any of the evaluations are NaN or infinite.
//
// If in a notebook, this would trigger a redraw of the plot.
//
// It implements [plot.Plotter]
func (pc *PlotConfig) DynamicSampleDone(incomplete bool) {
	if !incomplete {
		pc.pointsAdded++
	}
	if gonbui.IsNotebook && pc.gonbId != "" {
		pc.DynamicPlot(false)
	}
}

// Plot all figures with current data.
// If not in a notebook, this is a no-op.
func (pc *PlotConfig) Plot() error {
	if !gonbui.IsNotebook {
		return nil
	}
	for _, metricType := range slices.SortedKeys(pc.metricsTypesToFig) {
		figIdx := pc.metricsTypesToFig[metricType]
		gonbui.DisplayHtmlf("<p><b>Metric: %s</b></p>\n", metricType)
		err := gonbplotly.DisplayFig(pc.figs[figIdx])
		if err != nil {
			return err
		}
	}
	return nil
}

// DynamicPlot is called everytime a new metrics comes in, if configured for dynamic updates.
// Usually, not called directly by the user. Simply use [Dynamic] and schedule updates and this function
// will be called automatically.
//
// If `final` is true, it clears the transient area, and it plots instead in the definitive version.
func (pc *PlotConfig) DynamicPlot(final bool) {
	if !gonbui.IsNotebook {
		return
	}
	if pc.gonbId == "" {
		return
	}
	if pc.pointsAdded < 3 {
		return
	}
	if final == true {
		gonbui.UpdateHtml(pc.gonbId, "")
		err := pc.Plot()
		if err != nil {
			klog.Errorf("Failed to plot: %+v", err)
		}
		return
	}

	// Plot transient version.
	//gonbui.UpdateHtml(pc.gonbId, pc.PlotToHTML())
	elementId := gonbui.UniqueId()
	gonbui.UpdateHtml(pc.gonbId, fmt.Sprintf("<div id=%q></div>", elementId))

	for _, metricType := range slices.SortedKeys(pc.metricsTypesToFig) {
		figIdx := pc.metricsTypesToFig[metricType]
		dom.Append(elementId, fmt.Sprintf("<p><b>Metric: %s</b></p>\n", metricType))
		err := gonbplotly.AppendFig(elementId, pc.figs[figIdx])
		if err != nil {
			klog.Errorf("Failed to plot: %+v", err)
		}
	}
	return
}