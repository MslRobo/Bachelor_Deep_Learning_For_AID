# Session configuration
This directory should contain the json files that will be used to run multiple tests in a sequence.
With the Run configuration (RunConfig) files defining parameters for each specific run and can contain multiple mp4 files to be analyzed
With the Session configuration (SessionConfig) files you can define multiple different run configurations to be executed sequentially.
The primary use of this setup is to be able to predefine tests that the software should run, and execute them without any operational inputs inbetween allowing the user to start the tests and check back for the results.
The general structure of each json file is rigid and needs to follow the specifics defined in the blueprints. With some exeptions like args in runConfig can be removed if argsOverride is false.

Within ./RunPresets there can be found several presets for different args that can be used instead of manually defining them in the runConfig.json file.