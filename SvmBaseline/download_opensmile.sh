#!/bin/bash

wget -O opensmile.tar.gz https://github.com/audeering/opensmile/releases/download/v3.0.0/opensmile-3.0-linux-x64.tar.gz

tar -xzvf opensmile.tar.gz

mv opensmile-3.0*/ opensmile

chmod +x opensmile/bin/SMILExtract

ls -la opensmile/bin/

rm opensmile.tar.gz
