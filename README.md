# ParaEMT with OEDISI

Since there are no subscriptions to depend on, this federate
can be tested standalone with two scripts running simultaneously

```
helics_broker -f 1
```

and

```
python ParaEMT.py
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## How to use this in a simulation

0. (Preferred) Copy or clone oedisi-template to the simulation folder.
1. Add the location of the component_definition.json to a components.json in your simulation
   folder along with a preferred federate name.
2. Either create a system.json or add to an existing one with static information and any links.

```
{
    "components": [
        ...,
        {
            "name": "unique_simulation_name",
            "type": "TypeNameDefinedInComponentsDict",
            "parameters": {
                "input_needed_at_startup": 31.415926535
            }
        }
    ],
    "links": [
        ...,
        {
            "source": "unique_simulation_name",
            "source_port": "appropriate_helics_pub",
            "target": "another_federate_which_wants_data",
            "target_port": "subscription"
        }
    ]
}
```
3. Build with `oedisi build --system system.json`. A folder called `build`
should be created.
4. Run with `oedisi run`.

For a more complete example along with multi-container settings, see https://github.com/openEDI/oedisi-example.
