# Robot Installation and Setup
**Home Robot Instatallation:** Follow the [home-robot installation instructions](https://github.com/leo20021210/home-robot/blob/main/docs/install_robot.md) to install home-robot on your Stretch robot.

**Copy Hab Stretch Folder:** Copy hab stretch folder from [home robot repo](https://github.com/facebookresearch/home-robot/tree/main/assets/hab_stretch) 
```
cd $OK-Robot/
cp -r home-robot/assets/hab_stretch/ ok-robot-hw
```

## OpenAI installation

If you want to use the GPT-client to interact with OK-Robot in a conversational manner, you will need to install the OpenAI APIs.
Run this command to install the OpenAI API client:
```
pip install openai
```

Then, on the OpenAI website, create an API key and set it as an environment variable:
```
export OPENAI_API_KEY="your-api-key"
```