# Samples

Samples were created using MG5_aMC@NLO + Pythia 8 + Delphes. The `aMC` program was used to run all three parts.

## Available Samples
Generated samples are available on cori. Each version comes with a README with notes on any changes.

```shell
${CFS}/kkrizka/E290/samples/v0.0.1
```

The following samples should be there.
| **Name** | **Process** | **Size** | **Settings** | **Description**   |
|----------|-------------|----------|--------------|-------------------|
| dijet    | p p > j j   | 1M       | xptj > 400   | Background sample |
| ttbar    | p p > t t~  | 100k     | xptj > 400   | Example signal    |

## Instructions For New Samples

## Image
The MCProd image contains all the necessary programs for the sample creation.

```shell
shifter --image ghcr.io/kkrizka/mcprod:main
```

## Generating New Samples
Instructions for generating the dijet sample. They can be used as a template for a new sample.

1. Create a new sample and generate a gridpack. The gridpack won't be used later. It is only a convenient way to set card parmeters for the given process.
```shell
/opt/MG5aMC/3.3.1/bin/mg5_aMC ttbar.cmd
```

2. Generate 1M events with Pythia 8 shower and Delphes detector simulation.
```shell
cd PROC_ttbar
./bin/madevent ../run.cmd
```
