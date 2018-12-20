import sys


def print_progress(pre_str, pre_float, progress, last_str, last_float):
  """
  This function draw an active progress bar.

  usage:

  ```python
  from tools.progress_bar import print_progress
  import time

  print("asdfjlasdjfl")
  pre_str = "测试前缀"
  last_str = "测试后缀"
  total = 10

  for i in range(total):
  progress = float(i+1) / total
  print_progress(
    pre_str=pre_str, pre_float=float(i), progress=progress, last_str=last_str,
    last_float=progress
  )
  time.sleep(1)
  ```

  :param pre_str: 输出的前缀
  :param progress_float: 当前进度值
                       type: float
                       value: [0,1]
  :param last_str: 显示的后缀字符

  :return: Progressing bar
  """

  # Define the length of bar
  barLength = 30

  # Ceck the input!
  assert type(progress) is float, "id is not a float: %r" % id
  assert 0 <= progress <= 1, "variable should be between zero and one!"

  # Empty status while processing.
  status = ""

  # This part is to make a new line when the process is finished.
  if progress >= 1:
    progress = 1
    status = "\r\n"

  # Where we are in the progress!
  indicator = int(round(barLength * progress))

  # Print the appropriate progress phase!
  list = [pre_str, pre_float, "#" * indicator, "-" * (barLength - indicator),
          progress * 100, last_str, last_float, status]
  text = "\r{0[0]} {0[1]} {0[2]} {0[3]} %{0[4]:.2f} \
  {0[5]}={0[6]:.2f} {0[7]}".format(list)
  sys.stdout.write(text)
  sys.stdout.flush()
