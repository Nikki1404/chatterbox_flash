(tts_env) PS C:\Users\re_nikitav\Documents\chatterbox_flash> python .\client.py --url ws://127.0.0.1:8000/ws/tts --text-file C:\Users\re_nikitav\Documents\chatterbox_flash\call1.txt --output call1.wav --play
[INFO] Text file  : C:\Users\re_nikitav\Documents\chatterbox_flash\call1.txt
[INFO] Characters : 1411

======================================================================
CHATTERBOX FLASH - WEBSOCKET CLIENT
======================================================================
Server     : ws://127.0.0.1:8000/ws/tts
Characters : 1411
Output     : call1.wav
Text       : Hello, thank you for calling Inspira Financial for this call. Would you prefer to continue in English or Spanish language?  English please. Thank you ...
======================================================================

[1] Connecting to WebSocket...
[OK] Connected in 686.57 ms
[2] TTS request sent
[3] Waiting for audio...
[START] sample_rate=24000 Hz | channels=1
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Documents\chatterbox_flash\client.py", line 683, in <module>
    main()
    ~~~~^^
  File "C:\Users\re_nikitav\Documents\chatterbox_flash\client.py", line 672, in main
    asyncio.run(
    ~~~~~~~~~~~^
        run_tts(
        ^^^^^^^^
    ...<4 lines>...
        )
        ^
    )
    ^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Documents\chatterbox_flash\client.py", line 310, in run_tts
    raise RuntimeError(
        error_message
    )
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.