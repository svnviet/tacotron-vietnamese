import argparse
import falcon
from hparams import hparams, hparams_debug_string
import os
from synthesizer import Synthesizer
from pydub import AudioSegment
html_body = '''<html><title>Demo</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <textarea row="8" column="60" id="text"></textarea>
  <textarea row="8" column="60" id="speed"></textarea>
  <button id="button" name="synthesize">Speak</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = 'Synthesizing...'
    q('#button').disabled = true
    q('#audio').hidden = true
    synthesize(text)
  }
  e.preventDefault()
  return false
})


function synthesize(text) {
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.blob()
    }).then(function(blob) {
      q('#message').textContent = ''
      q('#button').disabled = false
      q('#audio').src = URL.createObjectURL(blob)
      q('#audio').hidden = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}
</script></body></html>
'''


class UIResource:
    def on_get(self, req, res):
        res.content_type = 'text/html'
        res.body = html_body


class SynthesisResource:
    def on_get(self, req, res):
        if not req.params.get('text'):
            raise falcon.HTTPBadRequest()
        count=0
        Rc=(r'.')
        ch=(r':;()=+_-')
        sentences=(req.params.get('text').split('.'))

        base_path =r'D:\tacotron_tensorflow-master\tacotron\testckpt\e\\'
        for i,text in enumerate(sentences):

            for k in ch:
                text= text.replace(k,',')

            for j in Rc:
                text=text.replace(j,'')


            for r_text in text.split(','):
                if len(r_text)>=1:
                    count+=1
                    path = '%s%04d.wav' % (base_path,count )

                    with open(path, 'wb') as f:
                        f.write(synthesizer.synthesize(r_text))
        Dir = os.listdir(base_path)
        Audio_resul=AudioSegment.silent()
        for i in (Dir):
            sound_synthe = AudioSegment.from_wav(base_path+i) + AudioSegment.silent(duration=350)
            Audio_resul +=sound_synthe
        # speed=1.5
        # Audio_resul = Audio_resul._spawn(Audio_resul.raw_data, overrides={
        #              "frame_rate": int(Audio_resul.frame_rate * speed)})
        # Audio_resul = Audio_resul.set_frame_rate(Audio_resul.frame_rate)
        # Audio_resul.export("path10.wav", format="wav")


        for i in (Dir):
            if os.path.exists(base_path+i):
                os.remove(base_path+i)
            else:
                print("Can not delete the file as it doesn't exists")
        with open("path10.wav",'rb') as f:
            res.data = f.read()
            res.content_type = 'audio/wav'


synthesizer = Synthesizer()
api = falcon.API()
api.add_route('/synthesize', SynthesisResource())
api.add_route('/', UIResource())


if __name__ == '__main__':
  from wsgiref import simple_server
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default=r'D:\tacotron_tensorflow-master\tacotron\testckpt\model.ckpt-366000')
  parser.add_argument('--port', type=int, default=9000)
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
  hparams.parse(args.hparams)
  print(hparams_debug_string())
  synthesizer.load(args.checkpoint)
  print('Serving on port %d' % args.port)
  simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
else:
  synthesizer.load(os.environ['CHECKPOINT'])
