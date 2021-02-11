import lsst.afw.table as afw_table
import lsst.log
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
    NoiseReplacerConfig,
    NoiseReplacer,
)
# import registers the CModel algorithm with meas_base
import lsst.meas.modelfit


def detect_and_deblend(*, exp, log):

    log = lsst.log.Log.getLogger("LSSTMEDSifier")

    thresh = 5.0
    loglevel = 'INFO'

    # This schema holds all the measurements that will be run within the
    # stack It needs to be constructed before running anything and passed
    # to algorithms that make additional measurents.
    schema = afw_table.SourceTable.makeMinimalSchema()

    # Setup algorithms to run
    meas_config = SingleFrameMeasurementConfig()
    meas_config.plugins.names = [
        "base_SdssCentroid",
        "base_PsfFlux",
        "base_SkyCoord",
        # "modelfit_ShapeletPsfApprox",
        "modelfit_DoubleShapeletPsfApprox",
        "modelfit_CModel",
        # "base_SdssShape",
        # "base_LocalBackground",
    ]

    # set these slots to none because we aren't running these algorithms
    meas_config.slots.apFlux = None
    meas_config.slots.gaussianFlux = None
    meas_config.slots.calibFlux = None
    meas_config.slots.modelFlux = None

    # goes with SdssShape above
    meas_config.slots.shape = None

    # fix odd issue where it things things are near the edge
    meas_config.plugins['base_SdssCentroid'].binmax = 1

    meas_task = SingleFrameMeasurementTask(
        config=meas_config,
        schema=schema,
    )

    # setup detection config
    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    detection_config.thresholdValue = thresh
    detection_task = SourceDetectionTask(config=detection_config)
    detection_task.log.setLevel(getattr(lsst.log, loglevel))

    deblend_config = SourceDeblendConfig()
    deblend_task = SourceDeblendTask(config=deblend_config, schema=schema)
    deblend_task.log.setLevel(getattr(lsst.log, loglevel))

    # Detect objects
    table = afw_table.SourceTable.make(schema)
    result = detection_task.run(table, exp)
    sources = result.sources

    # run the deblender
    deblend_task.run(exp, sources)

    # Run on deblended images
    noise_replacer_config = NoiseReplacerConfig()
    footprints = {
        record.getId(): (record.getParent(), record.getFootprint())
        for record in result.sources
    }

    # This constructor will replace all detected pixels with noise in the
    # image
    replacer = NoiseReplacer(
        noise_replacer_config,
        exposure=exp,
        footprints=footprints,
    )

    nbad = 0
    ntry = 0
    kept_sources = []

    for record in result.sources:

        # Skip parent objects where all children are inserted
        if record.get('deblend_nChild') != 0:
            continue

        ntry += 1

        # This will insert a single source into the image
        replacer.insertSource(record.getId())    # Get the peak as before

        # peak = record.getFootprint().getPeaks()[0]

        # The bounding box will be for the parent object
        # bbox = record.getFootprint().getBBox()

        meas_task.callMeasure(record, exp)

        # Remove object
        replacer.removeSource(record.getId())

        if record.getCentroidFlag():
            nbad += 1

        kept_sources.append(record)

    # Insert all objects back into image
    replacer.end()

    if ntry > 0:
        log.debug('nbad center: %d frac: %d' % (nbad, nbad/ntry))

    nkeep = len(kept_sources)
    ntot = len(result.sources)
    log.debug('kept %d/%d non parents' % (nkeep, ntot))
    return kept_sources
