# Install python3-venv
`sudo apt install python3-venv`

# clone weightAPI repo
use `git clone https://github.com/abhay-wyr/weightAPI.git` to clone the repo on aws instance 

open `cd weightAPI/` to open folder 

### create virtual envoirment in the clone folder 
`python -m venv <venv_name>

### activate vertual envoirment 
`source /<venv_name>/bin/acitvate`

###now install requirments.txt
`pip3 install pillow pandas pytesseract 
for opencv-python run this command `sudo apt-get install libgl1-mesa-glx`
Now try to run
`sudo apt-get install tesseract-ocr`




## For Gunicorn 
### Install Gunicorn 
`pip3 install 
### Reload systemd to apply the changes
`sudo systemctl daemon-reload`

### Start the service
`sudo systemctl start tableAPI`

### Enable the service to start on boot
`sudo systemctl enable tableAPI`

`
>[Unit]
>Description=Gunicorn instance for a table api <br/>
>After=network.target
>[Service]__
User=ubuntu__
Group=www-data__
WorkingDirectory=/home/ubuntu/tableAPI
ExecStart=/home/ubuntu/tableAPI/venv_tableAPI/bin/python /home/ubuntu/tableAPI/venv_tableAPI/bin/gunicorn -b 0.0.0.0:8000 app:app
Restart=always
[Install]
WantedBy=multi-user.target
`

