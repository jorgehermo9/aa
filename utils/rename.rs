use std::fs;

fn main() {
    let path_str = "/home/david/downloads/aa/Samples";
    let src_path = fs::read_dir(path_str).unwrap();

    for file in src_path {
      let file_path = &file.as_ref().unwrap().path().into_os_string().into_string().unwrap();
      let mut file_name = file.as_ref().unwrap().file_name().into_string().unwrap();
      file_name = file_name.replace('"', "");

      let mut components: Vec<&str> = file_name.split("-").collect();

      components.remove(0);
      let note = components.remove(0);

      let mut note_num = 1;
      let mut simple_note = "".to_string();
      for ch in note.chars() {
        match ch.to_digit(10) {
          Some(i) => note_num = i+2,
          None => simple_note.push(ch)
        }
      }

      let mut new_name = format!("Piano-{}{}", simple_note, note_num);
      for elem in components.into_iter() {
        if elem.len() > 1 {
          new_name.push_str(&format!("-{}", elem))
        }
      }
      
      fs::rename(file_path, format!("/home/david/downloads/newSamples/{}", new_name));
    }
}
