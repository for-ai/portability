# portability

For most environments we use `poetry`

The base install command is

`poetry install`

However we have different commands depending on your environment

## M1

to install use:
`poetry run install --with m1`

## Intel

to install use:
`poetry run install --with intel`

## Virtual Machine

to install use:
`poetry run install --with vm`

## Running the frequency code

`poetry run python main.py`

## Running the tests

The tests require an environment variable specifying the correct device. In most environments you can run the tests with the following command

`DEVICE=$device poetry run pytest`

where $device would be the different devices cpu, gpu, tpu

So on a local machine the command would be:
`DEVICE=cpu poetry run pytest`

And on a gpu VM it would be:

`DEVICE=gpu poetry run pytest`

However due to the quirks of the TPU vms we are not using poetry. So there the command would be:
`DEVICE=tpu pytest`

If in any environment you want to see print commands you add you can append -s on your command like so:
`DEVICE=cpu poetry run pytest -s`
