#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.2),
    on Tue Nov 28 22:32:47 2023
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from init_code
# Datalogging
import pandas as pd
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.2'
expName = '186c experiment'  # from the Builder filename that created this script
expInfo = {
    'participant': randint(100, 999),
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/muhanz/Documents/GitHub/prosody_learning/186c experiment_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1512, 982], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "intro" ---
    intro_text = visual.TextStim(win=win, name='intro_text',
        text='Welcome to the experiment! \n\nThis experiment will require you to have headphones, so please have them on. \n\nYou will first have a few survey questions to answer. Please answer to the best of your ability! \n\nWhen you are ready, press "right arrow". \n\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_key_resp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from init_code
    # Declare number of training/trials beforehand
    NUM_TRAINING = 1            # max 6
    NUM_SAMPLES = 1             # max 2
    NUM_SIGNAL = 2              # max 16
    NUM_NOISE_SIMILAR = 1       # max 8
    NUM_NOISE_DIFFERENT = 1     # max 8
    NUM_TRIALS = NUM_SIGNAL + NUM_NOISE_SIMILAR + NUM_NOISE_DIFFERENT
    
    # Define PATHs
    PATH_FORM_LIST = 'src/186c experiment survey form.csv'
    
    PATH_TRAINING_LIST = 'src/training_list.csv'
    PATH_TRAINING_AUDIO = 'audio'
    
    PATH_SAMPLE_LIST = 'src/sample_list.csv'
    PATH_SAMPLE_AUDIO = 'audio'
    
    PATH_TRIAL_LIST = 'src/trial_list.csv'
    PATH_TRIAL_AUDIO = 'audio'
    
    PATH_FORM_DATA = f'experiment_data/{expInfo["participant"]}_form.csv'
    PATH_EXPERIMENT_DATA = f'experiment_data/{expInfo["participant"]}_data.csv'
    
    # Defines Participant's Condition
    isMono = expInfo['participant']
    
    # Initial Processing
    isMono = isMono % 2 == 1
    # Note: names in condition_list corresponds to the column names in csv
    condition_list = ['rhythmic', 'monotone']
    condition = condition_list[isMono]
    
    
    # Datalogging
    df = pd.DataFrame({'Signal': pd.Series(dtype='int'),
                       'Response': pd.Series(dtype='int'),
                       'Duration': pd.Series(dtype='float')})
    
    
    # Coding Test Realm
    #NUM_TRAINING = NUM_TRAINING if len(data.importConditions(PATH_TRAINING_LIST)) > NUM_TRAINING else len(data.importConditions(PATH_TRAINING_LIST))
    
    # --- Initialize components for Routine "presurvey_questions" ---
    win.allowStencil = True
    form = visual.Form(win=win, name='form',
        items=PATH_FORM_LIST,
        textHeight=0.03,
        font='Open Sans',
        randomize=False,
        style='custom...',
        fillColor='white', borderColor='transparent', itemColor='black', 
        responseColor='black', markerColor='black', colorSpace='rgb', 
        size=(1, 1),
        pos=(0, 0),
        itemPadding=0.05,
        depth=-1
    )
    survey_button = visual.ButtonStim(win, 
        text='Done!', font='Arvo',
        pos=(0, -.45),
        letterHeight=0.05,
        size=(0.2, 0.1), borderWidth=0.0,
        fillColor='white', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=1.0,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='survey_button',
        depth=-2
    )
    survey_button.buttonClock = core.Clock()
    survey_text = visual.TextStim(win=win, name='survey_text',
        text='Please answer the following presurvey questions!\n\n\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "training_instr" ---
    training_text = visual.TextStim(win=win, name='training_text',
        text='For the first part, you will be listening to audio clips of an unspecified language. These audios will consist of short conversational phrases. \n\nMake sure to pay attention!\n\nPress "space" to start.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    training_start = keyboard.Keyboard()
    
    # --- Initialize components for Routine "training" ---
    # Run 'Begin Experiment' code from training_code
    imported_training_list = data.importConditions(PATH_TRAINING_LIST)
    training_list = list()
    for item in imported_training_list:
        training_list.append(f'{PATH_TRAINING_AUDIO}/{item[condition]}')
    shuffle(training_list)
    training_fixation_1 = visual.TextStim(win=win, name='training_fixation_1',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    training_fixation_2 = visual.TextStim(win=win, name='training_fixation_2',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    training_sound = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='training_sound')
    training_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "sample_instr" ---
    sample_start = visual.TextStim(win=win, name='sample_start',
        text='For the second part, you will be listening to more audio clips. \n\nHowever, these audio clips will be muffled, and your task will be to identify if the played clip is the same language you heard in the first part. \n\nYou will first be given a few sample clips to practice with.\n\nWhen it prompts you to answer, press key "1" to say yes and key "2" to say no.\n\nPress "space" to begin playing samples. ',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sample_key = keyboard.Keyboard()
    
    # --- Initialize components for Routine "sample_trial" ---
    # Run 'Begin Experiment' code from sample_code
    imported_sample_list = data.importConditions(PATH_SAMPLE_LIST)
    sample_list = list()
    for item in imported_sample_list:
        sample_list.append(f'{PATH_SAMPLE_AUDIO}/{item["path"]}')
    sample_sound = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='sample_sound')
    sample_sound.setVolume(1.0)
    sample_fixation = visual.TextStim(win=win, name='sample_fixation',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    sample_kb = keyboard.Keyboard()
    sample_text = visual.TextStim(win=win, name='sample_text',
        text='Do you think this is the language you learned?\n\n(This is a sample)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "trial_instr" ---
    trial_start = visual.TextStim(win=win, name='trial_start',
        text='Now begins the actual task:\n\nYou will be listening to more audio clips like the samples. \n\nYour task will be to identify if the played clip is the same language you heard in the first part. \n\nRemember, key "1" is for yes and key "2" is for no.\n\nPress "space" to begin.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    trial_key = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial" ---
    # Run 'Begin Experiment' code from trial_code
    imported_trial_list = data.importConditions(PATH_TRIAL_LIST)
    trial_list = list()
    for item in imported_trial_list:
        trial_list.append((f'{PATH_TRIAL_AUDIO}/{item["path"]}', item["type"]))
    shuffle(trial_list)
    trial_sound = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='trial_sound')
    trial_sound.setVolume(1.0)
    trial_fixation = visual.TextStim(win=win, name='trial_fixation',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    trial_kb = keyboard.Keyboard()
    trial_text = visual.TextStim(win=win, name='trial_text',
        text='Do you think this is the language you learned?',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "conclusion" ---
    conclusion_text = visual.TextStim(win=win, name='conclusion_text',
        text="You've finished the task!\n\nThank you for participating.",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "intro" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro.started', globalClock.getTime())
    intro_key_resp.keys = []
    intro_key_resp.rt = []
    _intro_key_resp_allKeys = []
    # keep track of which components have finished
    introComponents = [intro_text, intro_key_resp]
    for thisComponent in introComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_text* updates
        
        # if intro_text is starting this frame...
        if intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_text.frameNStart = frameN  # exact frame index
            intro_text.tStart = t  # local t and not account for scr refresh
            intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_text.started')
            # update status
            intro_text.status = STARTED
            intro_text.setAutoDraw(True)
        
        # if intro_text is active this frame...
        if intro_text.status == STARTED:
            # update params
            pass
        
        # *intro_key_resp* updates
        waitOnFlip = False
        
        # if intro_key_resp is starting this frame...
        if intro_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_key_resp.frameNStart = frameN  # exact frame index
            intro_key_resp.tStart = t  # local t and not account for scr refresh
            intro_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_key_resp.started')
            # update status
            intro_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_key_resp.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _intro_key_resp_allKeys.extend(theseKeys)
            if len(_intro_key_resp_allKeys):
                intro_key_resp.keys = _intro_key_resp_allKeys[-1].name  # just the last key pressed
                intro_key_resp.rt = _intro_key_resp_allKeys[-1].rt
                intro_key_resp.duration = _intro_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in introComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in introComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro.stopped', globalClock.getTime())
    # check responses
    if intro_key_resp.keys in ['', [], None]:  # No response was made
        intro_key_resp.keys = None
    thisExp.addData('intro_key_resp.keys',intro_key_resp.keys)
    if intro_key_resp.keys != None:  # we had a response
        thisExp.addData('intro_key_resp.rt', intro_key_resp.rt)
        thisExp.addData('intro_key_resp.duration', intro_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "presurvey_questions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('presurvey_questions.started', globalClock.getTime())
    # reset survey_button to account for continued clicks & clear times on/off
    survey_button.reset()
    # keep track of which components have finished
    presurvey_questionsComponents = [form, survey_button, survey_text]
    for thisComponent in presurvey_questionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "presurvey_questions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code
        if form.complete:
            survey_button.fillColor = 'blue'
            survey_button.color = 'white'
        else:
            survey_button.fillColor = 'white'
            survey_button.color = 'white'
        
        # *form* updates
        
        # if form is starting this frame...
        if form.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            form.frameNStart = frameN  # exact frame index
            form.tStart = t  # local t and not account for scr refresh
            form.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(form, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'form.started')
            # update status
            form.status = STARTED
            form.setAutoDraw(True)
        
        # if form is active this frame...
        if form.status == STARTED:
            # update params
            pass
        # *survey_button* updates
        
        # if survey_button is starting this frame...
        if survey_button.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            survey_button.frameNStart = frameN  # exact frame index
            survey_button.tStart = t  # local t and not account for scr refresh
            survey_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_button.started')
            # update status
            survey_button.status = STARTED
            survey_button.setAutoDraw(True)
        
        # if survey_button is active this frame...
        if survey_button.status == STARTED:
            # update params
            pass
            # check whether survey_button has been pressed
            if survey_button.isClicked:
                if not survey_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    survey_button.timesOn.append(survey_button.buttonClock.getTime())
                    survey_button.timesOff.append(survey_button.buttonClock.getTime())
                elif len(survey_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    survey_button.timesOff[-1] = survey_button.buttonClock.getTime()
                if not survey_button.wasClicked:
                    # run callback code when survey_button is clicked
                    if form.complete:
                        continueRoutine = False
        # take note of whether survey_button was clicked, so that next frame we know if clicks are new
        survey_button.wasClicked = survey_button.isClicked and survey_button.status == STARTED
        
        # *survey_text* updates
        
        # if survey_text is starting this frame...
        if survey_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            survey_text.frameNStart = frameN  # exact frame index
            survey_text.tStart = t  # local t and not account for scr refresh
            survey_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(survey_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'survey_text.started')
            # update status
            survey_text.status = STARTED
            survey_text.setAutoDraw(True)
        
        # if survey_text is active this frame...
        if survey_text.status == STARTED:
            # update params
            pass
        
        # if survey_text is stopping this frame...
        if survey_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > survey_text.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                survey_text.tStop = t  # not accounting for scr refresh
                survey_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'survey_text.stopped')
                # update status
                survey_text.status = FINISHED
                survey_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in presurvey_questionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "presurvey_questions" ---
    for thisComponent in presurvey_questionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('presurvey_questions.stopped', globalClock.getTime())
    # Run 'End Routine' code from code
    form_data = form.getData()
    form_return = list()
    for item in form_data:
        form_return.append(item['response'])
    form_return.append(isMono)
    
    form_df = pd.DataFrame([form_return],
                           columns = ['n_languages',
                                      'lang_list',
                                      'prof_list',
                                      'sum_proficiency',
                                      'isMono'])
    form_df.to_csv(PATH_FORM_DATA)
    form.addDataToExp(thisExp, 'rows')
    form.autodraw = False
    thisExp.addData('survey_button.numClicks', survey_button.numClicks)
    if survey_button.numClicks:
       thisExp.addData('survey_button.timesOn', survey_button.timesOn)
       thisExp.addData('survey_button.timesOff', survey_button.timesOff)
    else:
       thisExp.addData('survey_button.timesOn', "")
       thisExp.addData('survey_button.timesOff', "")
    # the Routine "presurvey_questions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "training_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('training_instr.started', globalClock.getTime())
    training_start.keys = []
    training_start.rt = []
    _training_start_allKeys = []
    # keep track of which components have finished
    training_instrComponents = [training_text, training_start]
    for thisComponent in training_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "training_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *training_text* updates
        
        # if training_text is starting this frame...
        if training_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            training_text.frameNStart = frameN  # exact frame index
            training_text.tStart = t  # local t and not account for scr refresh
            training_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(training_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'training_text.started')
            # update status
            training_text.status = STARTED
            training_text.setAutoDraw(True)
        
        # if training_text is active this frame...
        if training_text.status == STARTED:
            # update params
            pass
        
        # *training_start* updates
        waitOnFlip = False
        
        # if training_start is starting this frame...
        if training_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            training_start.frameNStart = frameN  # exact frame index
            training_start.tStart = t  # local t and not account for scr refresh
            training_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(training_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'training_start.started')
            # update status
            training_start.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(training_start.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(training_start.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if training_start.status == STARTED and not waitOnFlip:
            theseKeys = training_start.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _training_start_allKeys.extend(theseKeys)
            if len(_training_start_allKeys):
                training_start.keys = _training_start_allKeys[-1].name  # just the last key pressed
                training_start.rt = _training_start_allKeys[-1].rt
                training_start.duration = _training_start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in training_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "training_instr" ---
    for thisComponent in training_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('training_instr.stopped', globalClock.getTime())
    # check responses
    if training_start.keys in ['', [], None]:  # No response was made
        training_start.keys = None
    thisExp.addData('training_start.keys',training_start.keys)
    if training_start.keys != None:  # we had a response
        thisExp.addData('training_start.rt', training_start.rt)
        thisExp.addData('training_start.duration', training_start.duration)
    thisExp.nextEntry()
    # the Routine "training_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    training_loop = data.TrialHandler(nReps=NUM_TRAINING, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='training_loop')
    thisExp.addLoop(training_loop)  # add the loop to the experiment
    thisTraining_loop = training_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTraining_loop.rgb)
    if thisTraining_loop != None:
        for paramName in thisTraining_loop:
            globals()[paramName] = thisTraining_loop[paramName]
    
    for thisTraining_loop in training_loop:
        currentLoop = training_loop
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTraining_loop.rgb)
        if thisTraining_loop != None:
            for paramName in thisTraining_loop:
                globals()[paramName] = thisTraining_loop[paramName]
        
        # --- Prepare to start Routine "training" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('training.started', globalClock.getTime())
        # Run 'Begin Routine' code from training_code
        training_audio = training_list[training_loop.thisRepN]
        #training_audio = imported_training_list[training_loop.thisRepN]
        training_sound.setSound(training_audio, secs=7.0, hamming=True)
        training_sound.setVolume(1.0, log=False)
        training_sound.seek(0)
        # keep track of which components have finished
        trainingComponents = [training_fixation_1, training_fixation_2, training_sound]
        for thisComponent in trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 12.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *training_fixation_1* updates
            
            # if training_fixation_1 is starting this frame...
            if training_fixation_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                training_fixation_1.frameNStart = frameN  # exact frame index
                training_fixation_1.tStart = t  # local t and not account for scr refresh
                training_fixation_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_fixation_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_fixation_1.started')
                # update status
                training_fixation_1.status = STARTED
                training_fixation_1.setAutoDraw(True)
            
            # if training_fixation_1 is active this frame...
            if training_fixation_1.status == STARTED:
                # update params
                pass
            
            # if training_fixation_1 is stopping this frame...
            if training_fixation_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > training_fixation_1.tStartRefresh + 8.5-frameTolerance:
                    # keep track of stop time/frame for later
                    training_fixation_1.tStop = t  # not accounting for scr refresh
                    training_fixation_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'training_fixation_1.stopped')
                    # update status
                    training_fixation_1.status = FINISHED
                    training_fixation_1.setAutoDraw(False)
            
            # *training_fixation_2* updates
            
            # if training_fixation_2 is starting this frame...
            if training_fixation_2.status == NOT_STARTED and tThisFlip >= 8.5-frameTolerance:
                # keep track of start time/frame for later
                training_fixation_2.frameNStart = frameN  # exact frame index
                training_fixation_2.tStart = t  # local t and not account for scr refresh
                training_fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(training_fixation_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'training_fixation_2.started')
                # update status
                training_fixation_2.status = STARTED
                training_fixation_2.setAutoDraw(True)
            
            # if training_fixation_2 is active this frame...
            if training_fixation_2.status == STARTED:
                # update params
                pass
            
            # if training_fixation_2 is stopping this frame...
            if training_fixation_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > training_fixation_2.tStartRefresh + 3.5-frameTolerance:
                    # keep track of stop time/frame for later
                    training_fixation_2.tStop = t  # not accounting for scr refresh
                    training_fixation_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'training_fixation_2.stopped')
                    # update status
                    training_fixation_2.status = FINISHED
                    training_fixation_2.setAutoDraw(False)
            
            # if training_sound is starting this frame...
            if training_sound.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                training_sound.frameNStart = frameN  # exact frame index
                training_sound.tStart = t  # local t and not account for scr refresh
                training_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('training_sound.started', tThisFlipGlobal)
                # update status
                training_sound.status = STARTED
                training_sound.play(when=win)  # sync with win flip
            
            # if training_sound is stopping this frame...
            if training_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > training_sound.tStartRefresh + 7.0-frameTolerance:
                    # keep track of stop time/frame for later
                    training_sound.tStop = t  # not accounting for scr refresh
                    training_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'training_sound.stopped')
                    # update status
                    training_sound.status = FINISHED
                    training_sound.stop()
            # update training_sound status according to whether it's playing
            if training_sound.isPlaying:
                training_sound.status = STARTED
            elif training_sound.isFinished:
                training_sound.status = FINISHED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "training" ---
        for thisComponent in trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('training.stopped', globalClock.getTime())
        training_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-12.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed NUM_TRAINING repeats of 'training_loop'
    
    
    # --- Prepare to start Routine "sample_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('sample_instr.started', globalClock.getTime())
    sample_key.keys = []
    sample_key.rt = []
    _sample_key_allKeys = []
    # keep track of which components have finished
    sample_instrComponents = [sample_start, sample_key]
    for thisComponent in sample_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "sample_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *sample_start* updates
        
        # if sample_start is starting this frame...
        if sample_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sample_start.frameNStart = frameN  # exact frame index
            sample_start.tStart = t  # local t and not account for scr refresh
            sample_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sample_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'sample_start.started')
            # update status
            sample_start.status = STARTED
            sample_start.setAutoDraw(True)
        
        # if sample_start is active this frame...
        if sample_start.status == STARTED:
            # update params
            pass
        
        # *sample_key* updates
        waitOnFlip = False
        
        # if sample_key is starting this frame...
        if sample_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sample_key.frameNStart = frameN  # exact frame index
            sample_key.tStart = t  # local t and not account for scr refresh
            sample_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sample_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'sample_key.started')
            # update status
            sample_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(sample_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(sample_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if sample_key.status == STARTED and not waitOnFlip:
            theseKeys = sample_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _sample_key_allKeys.extend(theseKeys)
            if len(_sample_key_allKeys):
                sample_key.keys = _sample_key_allKeys[-1].name  # just the last key pressed
                sample_key.rt = _sample_key_allKeys[-1].rt
                sample_key.duration = _sample_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in sample_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "sample_instr" ---
    for thisComponent in sample_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('sample_instr.stopped', globalClock.getTime())
    # check responses
    if sample_key.keys in ['', [], None]:  # No response was made
        sample_key.keys = None
    thisExp.addData('sample_key.keys',sample_key.keys)
    if sample_key.keys != None:  # we had a response
        thisExp.addData('sample_key.rt', sample_key.rt)
        thisExp.addData('sample_key.duration', sample_key.duration)
    thisExp.nextEntry()
    # the Routine "sample_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    sample_loop = data.TrialHandler(nReps=NUM_SAMPLES, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='sample_loop')
    thisExp.addLoop(sample_loop)  # add the loop to the experiment
    thisSample_loop = sample_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSample_loop.rgb)
    if thisSample_loop != None:
        for paramName in thisSample_loop:
            globals()[paramName] = thisSample_loop[paramName]
    
    for thisSample_loop in sample_loop:
        currentLoop = sample_loop
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisSample_loop.rgb)
        if thisSample_loop != None:
            for paramName in thisSample_loop:
                globals()[paramName] = thisSample_loop[paramName]
        
        # --- Prepare to start Routine "sample_trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('sample_trial.started', globalClock.getTime())
        # Run 'Begin Routine' code from sample_code
        sample_audio = sample_list[sample_loop.thisRepN]
        sample_sound.setSound(sample_audio, secs=10.0, hamming=True)
        sample_sound.setVolume(1.0, log=False)
        sample_sound.seek(0)
        sample_kb.keys = []
        sample_kb.rt = []
        _sample_kb_allKeys = []
        # keep track of which components have finished
        sample_trialComponents = [sample_sound, sample_fixation, sample_kb, sample_text]
        for thisComponent in sample_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "sample_trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 15.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if sample_sound is starting this frame...
            if sample_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sample_sound.frameNStart = frameN  # exact frame index
                sample_sound.tStart = t  # local t and not account for scr refresh
                sample_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sample_sound.started', tThisFlipGlobal)
                # update status
                sample_sound.status = STARTED
                sample_sound.play(when=win)  # sync with win flip
            
            # if sample_sound is stopping this frame...
            if sample_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sample_sound.tStartRefresh + 10.0-frameTolerance:
                    # keep track of stop time/frame for later
                    sample_sound.tStop = t  # not accounting for scr refresh
                    sample_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sample_sound.stopped')
                    # update status
                    sample_sound.status = FINISHED
                    sample_sound.stop()
            # update sample_sound status according to whether it's playing
            if sample_sound.isPlaying:
                sample_sound.status = STARTED
            elif sample_sound.isFinished:
                sample_sound.status = FINISHED
            
            # *sample_fixation* updates
            
            # if sample_fixation is starting this frame...
            if sample_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sample_fixation.frameNStart = frameN  # exact frame index
                sample_fixation.tStart = t  # local t and not account for scr refresh
                sample_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sample_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sample_fixation.started')
                # update status
                sample_fixation.status = STARTED
                sample_fixation.setAutoDraw(True)
            
            # if sample_fixation is active this frame...
            if sample_fixation.status == STARTED:
                # update params
                pass
            
            # if sample_fixation is stopping this frame...
            if sample_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sample_fixation.tStartRefresh + 10.0-frameTolerance:
                    # keep track of stop time/frame for later
                    sample_fixation.tStop = t  # not accounting for scr refresh
                    sample_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sample_fixation.stopped')
                    # update status
                    sample_fixation.status = FINISHED
                    sample_fixation.setAutoDraw(False)
            
            # *sample_kb* updates
            waitOnFlip = False
            
            # if sample_kb is starting this frame...
            if sample_kb.status == NOT_STARTED and tThisFlip >= 10.0-frameTolerance:
                # keep track of start time/frame for later
                sample_kb.frameNStart = frameN  # exact frame index
                sample_kb.tStart = t  # local t and not account for scr refresh
                sample_kb.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sample_kb, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sample_kb.started')
                # update status
                sample_kb.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(sample_kb.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(sample_kb.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if sample_kb is stopping this frame...
            if sample_kb.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sample_kb.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    sample_kb.tStop = t  # not accounting for scr refresh
                    sample_kb.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sample_kb.stopped')
                    # update status
                    sample_kb.status = FINISHED
                    sample_kb.status = FINISHED
            if sample_kb.status == STARTED and not waitOnFlip:
                theseKeys = sample_kb.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                _sample_kb_allKeys.extend(theseKeys)
                if len(_sample_kb_allKeys):
                    sample_kb.keys = _sample_kb_allKeys[-1].name  # just the last key pressed
                    sample_kb.rt = _sample_kb_allKeys[-1].rt
                    sample_kb.duration = _sample_kb_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *sample_text* updates
            
            # if sample_text is starting this frame...
            if sample_text.status == NOT_STARTED and tThisFlip >= 10.0-frameTolerance:
                # keep track of start time/frame for later
                sample_text.frameNStart = frameN  # exact frame index
                sample_text.tStart = t  # local t and not account for scr refresh
                sample_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sample_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sample_text.started')
                # update status
                sample_text.status = STARTED
                sample_text.setAutoDraw(True)
            
            # if sample_text is active this frame...
            if sample_text.status == STARTED:
                # update params
                pass
            
            # if sample_text is stopping this frame...
            if sample_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sample_text.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    sample_text.tStop = t  # not accounting for scr refresh
                    sample_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sample_text.stopped')
                    # update status
                    sample_text.status = FINISHED
                    sample_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in sample_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "sample_trial" ---
        for thisComponent in sample_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('sample_trial.stopped', globalClock.getTime())
        sample_sound.pause()  # ensure sound has stopped at end of Routine
        # check responses
        if sample_kb.keys in ['', [], None]:  # No response was made
            sample_kb.keys = None
        sample_loop.addData('sample_kb.keys',sample_kb.keys)
        if sample_kb.keys != None:  # we had a response
            sample_loop.addData('sample_kb.rt', sample_kb.rt)
            sample_loop.addData('sample_kb.duration', sample_kb.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-15.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed NUM_SAMPLES repeats of 'sample_loop'
    
    
    # --- Prepare to start Routine "trial_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('trial_instr.started', globalClock.getTime())
    trial_key.keys = []
    trial_key.rt = []
    _trial_key_allKeys = []
    # keep track of which components have finished
    trial_instrComponents = [trial_start, trial_key]
    for thisComponent in trial_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *trial_start* updates
        
        # if trial_start is starting this frame...
        if trial_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            trial_start.frameNStart = frameN  # exact frame index
            trial_start.tStart = t  # local t and not account for scr refresh
            trial_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(trial_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'trial_start.started')
            # update status
            trial_start.status = STARTED
            trial_start.setAutoDraw(True)
        
        # if trial_start is active this frame...
        if trial_start.status == STARTED:
            # update params
            pass
        
        # *trial_key* updates
        waitOnFlip = False
        
        # if trial_key is starting this frame...
        if trial_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            trial_key.frameNStart = frameN  # exact frame index
            trial_key.tStart = t  # local t and not account for scr refresh
            trial_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(trial_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'trial_key.started')
            # update status
            trial_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(trial_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(trial_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if trial_key.status == STARTED and not waitOnFlip:
            theseKeys = trial_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _trial_key_allKeys.extend(theseKeys)
            if len(_trial_key_allKeys):
                trial_key.keys = _trial_key_allKeys[-1].name  # just the last key pressed
                trial_key.rt = _trial_key_allKeys[-1].rt
                trial_key.duration = _trial_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trial_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial_instr" ---
    for thisComponent in trial_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('trial_instr.stopped', globalClock.getTime())
    # check responses
    if trial_key.keys in ['', [], None]:  # No response was made
        trial_key.keys = None
    thisExp.addData('trial_key.keys',trial_key.keys)
    if trial_key.keys != None:  # we had a response
        thisExp.addData('trial_key.rt', trial_key.rt)
        thisExp.addData('trial_key.duration', trial_key.duration)
    thisExp.nextEntry()
    # the Routine "trial_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_loop = data.TrialHandler(nReps=NUM_TRIALS, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trial_loop')
    thisExp.addLoop(trial_loop)  # add the loop to the experiment
    thisTrial_loop = trial_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
    if thisTrial_loop != None:
        for paramName in thisTrial_loop:
            globals()[paramName] = thisTrial_loop[paramName]
    
    for thisTrial_loop in trial_loop:
        currentLoop = trial_loop
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
        if thisTrial_loop != None:
            for paramName in thisTrial_loop:
                globals()[paramName] = thisTrial_loop[paramName]
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime())
        # Run 'Begin Routine' code from trial_code
        trial_audio = trial_list[trial_loop.thisRepN][0]
        trial_sound.setSound(trial_audio, secs=10.0, hamming=True)
        trial_sound.setVolume(1.0, log=False)
        trial_sound.seek(0)
        trial_kb.keys = []
        trial_kb.rt = []
        _trial_kb_allKeys = []
        # keep track of which components have finished
        trialComponents = [trial_sound, trial_fixation, trial_kb, trial_text]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 15.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if trial_sound is starting this frame...
            if trial_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_sound.frameNStart = frameN  # exact frame index
                trial_sound.tStart = t  # local t and not account for scr refresh
                trial_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('trial_sound.started', tThisFlipGlobal)
                # update status
                trial_sound.status = STARTED
                trial_sound.play(when=win)  # sync with win flip
            
            # if trial_sound is stopping this frame...
            if trial_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_sound.tStartRefresh + 10.0-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_sound.tStop = t  # not accounting for scr refresh
                    trial_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_sound.stopped')
                    # update status
                    trial_sound.status = FINISHED
                    trial_sound.stop()
            # update trial_sound status according to whether it's playing
            if trial_sound.isPlaying:
                trial_sound.status = STARTED
            elif trial_sound.isFinished:
                trial_sound.status = FINISHED
            
            # *trial_fixation* updates
            
            # if trial_fixation is starting this frame...
            if trial_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_fixation.frameNStart = frameN  # exact frame index
                trial_fixation.tStart = t  # local t and not account for scr refresh
                trial_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_fixation.started')
                # update status
                trial_fixation.status = STARTED
                trial_fixation.setAutoDraw(True)
            
            # if trial_fixation is active this frame...
            if trial_fixation.status == STARTED:
                # update params
                pass
            
            # if trial_fixation is stopping this frame...
            if trial_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_fixation.tStartRefresh + 10.0-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_fixation.tStop = t  # not accounting for scr refresh
                    trial_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_fixation.stopped')
                    # update status
                    trial_fixation.status = FINISHED
                    trial_fixation.setAutoDraw(False)
            
            # *trial_kb* updates
            waitOnFlip = False
            
            # if trial_kb is starting this frame...
            if trial_kb.status == NOT_STARTED and tThisFlip >= 10.0-frameTolerance:
                # keep track of start time/frame for later
                trial_kb.frameNStart = frameN  # exact frame index
                trial_kb.tStart = t  # local t and not account for scr refresh
                trial_kb.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_kb, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_kb.started')
                # update status
                trial_kb.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(trial_kb.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(trial_kb.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if trial_kb is stopping this frame...
            if trial_kb.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_kb.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_kb.tStop = t  # not accounting for scr refresh
                    trial_kb.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_kb.stopped')
                    # update status
                    trial_kb.status = FINISHED
                    trial_kb.status = FINISHED
            if trial_kb.status == STARTED and not waitOnFlip:
                theseKeys = trial_kb.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                _trial_kb_allKeys.extend(theseKeys)
                if len(_trial_kb_allKeys):
                    trial_kb.keys = _trial_kb_allKeys[-1].name  # just the last key pressed
                    trial_kb.rt = _trial_kb_allKeys[-1].rt
                    trial_kb.duration = _trial_kb_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *trial_text* updates
            
            # if trial_text is starting this frame...
            if trial_text.status == NOT_STARTED and tThisFlip >= 10.0-frameTolerance:
                # keep track of start time/frame for later
                trial_text.frameNStart = frameN  # exact frame index
                trial_text.tStart = t  # local t and not account for scr refresh
                trial_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_text.started')
                # update status
                trial_text.status = STARTED
                trial_text.setAutoDraw(True)
            
            # if trial_text is active this frame...
            if trial_text.status == STARTED:
                # update params
                pass
            
            # if trial_text is stopping this frame...
            if trial_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_text.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_text.tStop = t  # not accounting for scr refresh
                    trial_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_text.stopped')
                    # update status
                    trial_text.status = FINISHED
                    trial_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime())
        # Run 'End Routine' code from trial_code
        new_row = pd.Series({'Signal': trial_list[trial_loop.thisRepN][1],
                             'Response': trial_kb.keys,
                             'Duration': trial_kb.rt})
        
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
        trial_sound.pause()  # ensure sound has stopped at end of Routine
        # check responses
        if trial_kb.keys in ['', [], None]:  # No response was made
            trial_kb.keys = None
        trial_loop.addData('trial_kb.keys',trial_kb.keys)
        if trial_kb.keys != None:  # we had a response
            trial_loop.addData('trial_kb.rt', trial_kb.rt)
            trial_loop.addData('trial_kb.duration', trial_kb.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-15.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed NUM_TRIALS repeats of 'trial_loop'
    
    
    # --- Prepare to start Routine "conclusion" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('conclusion.started', globalClock.getTime())
    # Run 'Begin Routine' code from conc_code
    df.to_csv(PATH_EXPERIMENT_DATA)
    # keep track of which components have finished
    conclusionComponents = [conclusion_text]
    for thisComponent in conclusionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "conclusion" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *conclusion_text* updates
        
        # if conclusion_text is starting this frame...
        if conclusion_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            conclusion_text.frameNStart = frameN  # exact frame index
            conclusion_text.tStart = t  # local t and not account for scr refresh
            conclusion_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(conclusion_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'conclusion_text.started')
            # update status
            conclusion_text.status = STARTED
            conclusion_text.setAutoDraw(True)
        
        # if conclusion_text is active this frame...
        if conclusion_text.status == STARTED:
            # update params
            pass
        
        # if conclusion_text is stopping this frame...
        if conclusion_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > conclusion_text.tStartRefresh + 10.0-frameTolerance:
                # keep track of stop time/frame for later
                conclusion_text.tStop = t  # not accounting for scr refresh
                conclusion_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'conclusion_text.stopped')
                # update status
                conclusion_text.status = FINISHED
                conclusion_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in conclusionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "conclusion" ---
    for thisComponent in conclusionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('conclusion.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
