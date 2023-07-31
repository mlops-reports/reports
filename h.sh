#!/bin/bash
heroku ps:scale web=$1 --app er-reports

if [ $1 -eq 1 ]; then
    open https://label.drgoktugasci.com/projects/ 
    exit 1
fi