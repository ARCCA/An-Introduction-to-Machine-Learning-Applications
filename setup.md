---
title: Setup
---

For this workshop you will need to:
-  In general we will work with datasets that come preinstalled with python libraries
   but for lesson 2 we will use a couple of additional datasets. You can download 
   them from [<ins>here</ins>]({{ page.root }}/files/lesson02-gdp-bli.zip) and [<ins>here</ins>]({{ page.root }}/files/pretrained_models_and_transfer_learning.zip)


-  Install Anaconda following the instructions for your platform in their official
   [documentation website](https://docs.anaconda.com/anaconda/install/).
-  Once Anaconda is installed, be sure to have the following libraries:
     -    python<=3.12
     -    keras
     -    matplotlib
     -    tensorflow=2
     -    ipykernel
     -    jupyter
     -    scikit-learn
     -    pandas
     -    pip
    
   Or if you prefer you can use [<ins>this yml file</ins>]({{ page.root }}/files/environment.yml) 
   to create a virtual environment with all the necessary libraries. You can use the
   following instructions to install the environment:



## Install Anaconda

{::options parse_block_html="true" /}
<div>
  <ul class="nav nav-tabs" role="tablist">
  <li role="presentation" class="active"><a data-os="windows" href="#Windows" aria-controls="Windows" role="tab" data-toggle="tab">Windows</a></li>
  <li role="presentation"><a data-os="macos" href="#Macos" aria-controls="Macos" role="tab" data-toggle="tab">Macos</a></li>
  <li role="presentation"><a data-os="linux" href="#Linux" aria-controls="Linux" role="tab" data-toggle="tab">Linux</a></li>
  </ul>
  
  <div class="tab-content">
  <article role="tabpanel" class="tab-pane active" id="Windows">
  Open Anaconda Navigator and click **Environments**
  <img src="{{ page.root }}/fig/setup-environment-windows-01.png" alt="setup-environment-windows-01" width="50%" height="50%" />

  At the bottom, select **Import** and search for the environment file we just
  downloaded (note that the name for the environment is automatically determined from
  the environment file, you can change this name by opening the file in a text 
  editor). Select the green **Import** button.
  <img src="{{ page.root }}/fig/setup-environment-windows-02.png" alt="setup-environment-windows-02" width="50%" height="50%" />

  Anaconda will start building your environment. This might take a few minutes.
  <img src="{{ page.root }}/fig/setup-environment-windows-03.png" alt="setup-environment-windows-03" width="50%" height="50%" />

  Once Anaconda has finished building the environment, use it to open a new Jupyter 
  Notebook.
  <img src="{{ page.root }}/fig/setup-environment-windows-04.png" alt="setup-environment-windows-04" width="50%" height="50%" />

  If everything went well, you should see a new window displaying your 
  Jupyter Notebook with access to the packages listed in our environment file. If you
  had any issues, please get in contact with us at arcca@cardiff.ac.uk.
  <img src="{{ page.root }}/fig/setup-environment-windows-05.png" alt="setup-environment-windows-05" width="50%" height="50%" />

  </article>
  
  <article role="tabpanel" class="tab-pane" id="Macos">
  Open Anaconda Navigator and click **Environments**
  <img src="{{ page.root }}/fig/setup-environment-macos-01.png" alt="setup-environment-macos-01" width="50%" height="50%" />

  At the bottom, select **Import** and search for the environment file we just
  downloaded (note that the name for the environment is automatically determined from
  the environment file, you can change this name by opening the file in a text 
  editor). Select the green **Import** button.
  <img src="{{ page.root }}/fig/setup-environment-macos-02.png" alt="setup-environment-macos-02" width="50%" height="50%" />

  Anaconda will start building your environment. This might take a few minutes.
  <img src="{{ page.root }}/fig/setup-environment-macos-03.png" alt="setup-environment-macos-03" width="50%" height="50%" />

  Once Anaconda has finished building the environment, use it to open a new Jupyter 
  Notebook.
  <img src="{{ page.root }}/fig/setup-environment-macos-04.png" alt="setup-environment-macos-04" width="50%" height="50%" />

  If everything went well, you should see a new window displaying your 
  Jupyter Notebook with access to the packages listed in our environment file. If you
  had any issues, please get in contact with us at arcca@cardiff.ac.uk.
  <img src="{{ page.root }}/fig/setup-environment-macos-05.png" alt="setup-environment-macos-05" width="50%" height="50%" />

  </article>
  
  <article role="tabpanel" class="tab-pane" id="Linux">
   In a terminal window, create a conda virtual environment using the environment 
   file.
   ~~~
   conda env create -f environment.yml
   ~~~
   {: .language-bash}

   and then activate your environment:
   ~~~
   conda activate machine-learning
   ~~~
   {: .language-bash}
   
   If you are comfortable working in the command line, you can directly access the 
   Python prompt:
   ~~~
   (machine-learning) user@user-host:~ $ python
   Python 3.7.9 (default, Aug 31 2020, 07:22:35)
   [Clang 10.0.0 ] :: Anaconda, Inc. on darwin
   Type "help", "copyright", "credits" or "license" for more information.
   >>>
   ~~~

   Alternatively, you can open a Jupyter Notebook tab with:
   ~~~
   (machine-learning) user@user-host:~ $ jupyter-notebook
   [I 12:29:10.612 NotebookApp] The port 8888 is already in use, trying another port.
   [I 12:29:10.618 NotebookApp] Serving notebooks from local directory: /Users/jose/git/An-Introduction-to-Machine-Learning-Applications/fig
   [I 12:29:10.618 NotebookApp] Jupyter Notebook 6.1.4 is running at:
   [I 12:29:10.618 NotebookApp] http://localhost:8889/?token=b5513abc7a4db29aa1655af6fffa0dc30b87c2eb2cd892c0
   [I 12:29:10.618 NotebookApp]  or http://127.0.0.1:8889/?token=b5513abc7a4db29aa1655af6fffa0dc30b87c2eb2cd892c0
   [I 12:29:10.619 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
   [C 12:29:10.629 NotebookApp]
   
       To access the notebook, open this file in a browser:
           file:///Users/jose/Library/Jupyter/runtime/nbserver-81468-open.html
       Or copy and paste one of these URLs:
           http://localhost:8889/?token=b5513abc7a4db29aa1655af6fffa0dc30b87c2eb2cd892c0
        or http://127.0.0.1:8889/?token=b5513abc7a4db29aa1655af6fffa0dc30b87c2eb2cd892c0
   [I 12:29:30.263 NotebookApp] Creating new notebook in
   [I 12:29:32.759 NotebookApp] Kernel started: 99950768-b57d-4440-b6cc-8b41814c37bf, name: python3
   [I 12:31:32.837 NotebookApp] Saving file at /Untitled.ipynb
   [I 12:32:20.493 NotebookApp] Starting buffering for 99950768-b57d-4440-b6cc-8b41814c37bf:9bba8bf12f03444b88af47393b6280dc
   ~~~
  If everything went well, you should see a new window displaying your 
  Jupyter Notebook with access to the packages listed in our environment file. If you
  had any issues, please get in contact with us at arcca@cardiff.ac.uk.
  <img src="{{ page.root }}/fig/setup-environment-linux-01.png" alt="setup-environment-linux-01" width="50%" height="50%" />
   {: .language-bash}

  </article>

  </div>
</div>


{% include links.md %}
