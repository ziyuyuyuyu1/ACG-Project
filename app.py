from flask import Flask, request, Response
import paramiko
import json

app = Flask(__name__)

@app.route('/run-command', methods=['POST'])
def run_command():
    command = request.form['command']
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('10.210.5.17', username='jialuo', password='jialuo2023')
    stdin, stdout, stderr = ssh.exec_command(command)
    result = stdout.read().decode()
    ssh.close()
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0')