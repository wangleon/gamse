Frequently Asked Questions
==========================

Software Running
----------------

**I have encountered a 'TkAgg' backend problem**

The error message:
::

    Something like ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running

is probably because you missed some libraries related to TK.
In Ubuntu, you can install these packages by running:

.. code-block:: bash

   apt-get install tk-dev libagg-dev

In RedHat/Fedora/CentOS, by running:

.. code-block:: bash

   yum install tk-devel agg agg-devel

Then, you have to reinstall matplotlib by running:

.. code-block:: bash

   pip --no-cache-dir install -U --force-reinstall matplotlib

You may need `sudo` in front of the above commands to get the authentication.

**Observing log file cannot be loaded correctly**

An error message like:
::
    ValueError: Input values did not match any of the formats where the format keyword is optional ['datetime', 'iso', 'isot', 'yday', 'datetime64', 'fits', 'byear_str', 'jyear_str']

is probably caused by unexpected strings and/or alignment problem in the
`.obslog` file.
For example, aligments with tab characters instead of space can result in this
problem.
