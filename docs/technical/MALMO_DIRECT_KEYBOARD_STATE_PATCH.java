// 修改 Malmo 的 CommandForKey.java，直接操作 Minecraft KeyboardListener
// 文件位置: /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build/sources/main/java/com/microsoft/Malmo/MissionHandlers/CommandForKey.java

// 在 CommandForKey 类中添加新方法：

import net.minecraft.client.KeyboardListener;
import net.minecraft.client.Minecraft;
import java.util.ArrayList;
import java.util.List;
import java.util.HashSet;
import java.util.Set;

public class CommandForKey extends CommandBase {
    
    // 现有代码...
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 新增：直接状态管理（MCP-Reborn 风格）
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    // 跟踪当前应该按下的键
    private static Set<String> currentPressedKeys = new HashSet<>();
    
    /**
     * 直接设置 Minecraft 键盘状态（不通过 FakeKeyboard）
     * 
     * @param keyName Minecraft 键名，例如 "key.keyboard.e"
     * @param pressed true=按下，false=释放
     */
    private void setKeyStateDirect(String keyName, boolean pressed) {
        if (pressed) {
            currentPressedKeys.add(keyName);
        } else {
            currentPressedKeys.remove(keyName);
        }
        
        // 创建新的键盘状态
        KeyboardListener.State newState = new KeyboardListener.State(
            new ArrayList<>(currentPressedKeys),  // 当前按下的键
            new ArrayList<>(),                     // 新按下的键（空）
            ""                                     // 字符输入（空）
        );
        
        // 直接设置到 Minecraft
        Minecraft mc = Minecraft.getInstance();
        if (mc != null && mc.keyboardListener != null) {
            // 这需要 Minecraft 暴露一个方法来接受外部状态
            // 类似 MCP-Reborn 的 PlayRecorder.setMouseKeyboardState()
            mc.keyboardListener.setState(newState);
        }
    }
    
    /**
     * 修改后的 execute 方法，使用直接状态管理
     */
    @Override
    public boolean execute(String verb, String parameter) {
        // 将命令动作映射到 Minecraft 键名
        String keyName = actionToKeyName(verb);
        
        if (keyName != null) {
            boolean pressed = !parameter.equalsIgnoreCase("0");
            
            System.out.println("[CommandForKey] Direct state: " + keyName + " = " + pressed);
            
            // 使用直接状态管理，而不是 FakeKeyboard
            setKeyStateDirect(keyName, pressed);
            
            return true;
        }
        
        // 回退到原始实现
        return super.execute(verb, parameter);
    }
    
    /**
     * 将 Malmo 动作映射到 Minecraft 键名
     */
    private String actionToKeyName(String action) {
        switch (action.toLowerCase()) {
            case "forward": return "key.keyboard.w";
            case "back": return "key.keyboard.s";
            case "left": return "key.keyboard.a";
            case "right": return "key.keyboard.d";
            case "inventory": return "key.keyboard.e";
            case "jump": return "key.keyboard.space";
            case "sneak": return "key.keyboard.left.shift";
            case "drop": return "key.keyboard.q";
            case "use": return "key.mouse.right";
            case "attack": return "key.mouse.left";
            default: return null;
        }
    }
}

/* 
 * 使用说明：
 * 
 * 1. 备份原始文件：
 *    cp CommandForKey.java CommandForKey.java.backup
 * 
 * 2. 修改 net/minecraft/client/KeyboardListener.java，添加 setState 方法：
 *    
 *    public class KeyboardListener {
 *        private State currentState;
 *        
 *        // 新增方法
 *        public void setState(State newState) {
 *            this.currentState = newState;
 *            // 更新 Minecraft 的键绑定状态
 *            for (String key : newState.keys) {
 *                KeyBinding binding = getKeyBindingByName(key);
 *                if (binding != null) {
 *                    KeyBinding.setKeyBindState(binding.getKey(), true);
 *                }
 *            }
 *        }
 *    }
 * 
 * 3. 重新编译 Malmo JAR：
 *    cd /path/to/Malmo/Minecraft
 *    ./gradlew clean shadowJar
 * 
 * 预期效果：
 * - inventory=1 → 打开 GUI
 * - inventory=0 → GUI 保持打开（不触发 release 事件）
 * - 完全绕过 FakeKeyboard 的事件机制
 */


